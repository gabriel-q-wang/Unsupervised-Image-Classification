import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch
import torch.nn as nn
import os
import utils
import networks
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from tensorboardX import SummaryWriter
from losses import SupervisedLoss
from custom_dataset import AugmentedCIFAR10, Sobel

class Model():
    def __init__(self, config_path):
        self.config_path = config_path
        config = utils.load_yaml(config_path)
        self.config = config
        self.device_ids = [int(x) for x in config["device"].split(':')[-1].split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(_id) for _id in self.device_ids)
        self.device = torch.device("cuda" if config["use_cuda"] else "cpu")

        # Given loss type, how many input images do we use
        # for supervised we use rgb, for unsupervised we use gray-scale
        self.loss_type_dict = {
            "supervised": 3,
            "unsupervised": 1
        }

        self.model = self.build_model()
        self.sobel = Sobel().to(self.device)

        if config["load_weights_folder"] is not None:
            self.load_model()

    def build_model(self):
        """
        Sets model to be whatever neural network we want with specified
        num_classes and whether or not we want to use pretrained imagenet weights
        """

        network_name = self.config["network_name"]
        num_classes = self.config["num_classes"]
        pretrained = self.config["pretrained_imagenet"]
        loss_type = self.config["loss_type"]
        self.num_images = self.loss_type_dict[loss_type]

        # TODO: Need to fix resnet; currently does not work
        if "resnet" in network_name:
            num_layers = int(network_name.split("resnet")[1])
            model = networks.resnet(num_layers, num_classes, pretrained)
        elif network_name == "alexnet":
            model = networks.alexnet(self.num_images, num_classes, pretrained)
        else:
            raise RuntimeError("%s is an invalid network_name" % network_name)

        model = model.to(self.device)
        return model

    def build_dataloaders(self, use_all=False):
        dataset_name = self.config["dataset_name"]
        dataset_path = self.config["dataset_path"]
        train_batch_size = self.config["batch_size"]
        val_batch_size = train_batch_size
        num_workers = self.config["num_workers"]

        dataset_dict = {
            "cifar10": AugmentedCIFAR10
        }

        print("Building Dataloaders...")
        train_dataset = dataset_dict[dataset_name](root=dataset_path, train=True, download=True)
        val_dataset = dataset_dict[dataset_name](root=dataset_path, train=False, download=True)
        class_names = train_dataset.classes

        if use_all:
            train_batch_size = len(train_dataset)
            val_batch_size = len(val_dataset)

        train_loader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        return train_loader, val_loader, class_names
        

    def set_mode(self, mode):
        if mode == "train":
            self.is_train = True
        elif mode == "eval":
            self.is_train = False
        else:
            raise RuntimeError("Invalid mode %s" % mode)
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

    def log_time(self, duration, loss):
        """Print a logging statement to the terminal
        """
        batch_size = self.config["batch_size"]
        samples_per_sec = batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.batch_idx, samples_per_sec, loss,
                                  utils.sec_to_dhm_str(time_sofar),
                                  utils.sec_to_dhm_str(training_time_left)))

    def log(self, images, y_hat, y, loss, mode):
        """
        Log to tensorbard
        """
        writer = self.writers[mode]        

        writer.add_scalar("Loss", loss, self.step)

        # Get predictions and gt class names as strings
        preds = self.logit2class(y_hat, is_gt=False)
        targets = self.logit2class(y, is_gt=True)

        for i in range(min(4, images.shape[0])):
            image = images[i]
            pred = preds[i]
            target = targets[i]
            image = image.permute(1,2,0).detach().cpu().numpy()
            image *= 255
            image = image.astype(np.uint8)
            # Need to do this conversion, otherwise can't use cv2.putText
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height,width = image.shape[0],image.shape[1]
            # Need to enlarge the image; otherwise we can't fit text on image
            image = cv2.resize(image, (32*5,32*5))
            height,width = image.shape[0],image.shape[1]
            text = "pred: {} | gt: {}".format(pred, target)
            y_loc = int(0.05*height) + 1
            x_loc = int(0.05*width) + 1
            cv2.putText(image, text, org=(x_loc, y_loc), fontFace=1, fontScale=0.5, color=(0,255,0), thickness=1)
            # Don't forget to change back to RGB so image is displayed properly on tensorboard
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(2,0,1)
            writer.add_image("{}/color_{}".format(mode, i), image, self.step)

    def logit2class(self, y, is_gt=False):
        # Class with largest positive logit value is chosen as predicted class
        if not is_gt:
            indexes = torch.argmax(y, dim=1)
        else:
            indexes = y
        class_names = [self.class_names[index.item()] for index in indexes]
        return class_names

    def process_inputs(self, inputs):
        new_inputs = []
        for _input in inputs:
            new_inputs.append(_input.to(self.device))
        image, y = new_inputs[0], new_inputs[1]
        if self.num_images == 3:
            input_image = image
        elif self.num_images == 1:
            input_image = self.sobel(image)
        else:
            raise RuntimeError("invalid number of input images %d" % self.num_images)
        # image, y = inputs[0], inputs[1]
        # image = image.to(self.device)
        # y = y.to(self.device)
        return input_image, y

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        model_name = self.config["network_name"]
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)
    
    def load_model(self):
        """
        Load model weights from disk
        """

        load_folder = self.config["load_weights_folder"]
        assert os.path.isdir(load_folder), \
            "Cannot find folder {}".format(load_folder)
        print("loading model from folder {}".format(load_folder))
        path = os.path.join(load_folder, "{}.pth".format(self.config["network_name"]))

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


    def run_epoch(self):

        for batch_idx, inputs in enumerate(self.train_loader): 
            input_image, y = self.process_inputs(inputs)
            start_time = time.time()
            self.batch_idx = batch_idx

            # Feed-forward image through neural net
            y_hat = self.model.forward(input_image)
            # Compute the loss
            loss = self.loss_fn.forward(y_hat, y)
            # Perform optimization step; backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            duration = time.time() - start_time

            log_frequency = self.config["log_frequency"]
            batch_size = self.config["batch_size"]

            # How many steps to wait before switching from frequent tensorboard
            # logging to less frequent tensorboard logging
            # set to how many ever steps are in one epoch
            step_cutoff = len(self.train_loader)

            early_phase = self.step % log_frequency == 0 and self.step < step_cutoff
            # Once past first epoch, only log every epoch
            late_phase = self.step % step_cutoff == 0

            if early_phase or late_phase:
                # Log time, tensorboard, and evaluate on validation set
                self.log_time(duration, loss)
                self.log(input_image, y_hat, y, loss, mode="train")
                self.val()

            self.step += 1


    def train(self):
        self.set_mode("train")
        # if self.config["load_weights_folder"] is not None:
        #     self.load_model()


        log_dir = self.config["log_dir"]
        dataset_name = self.config["dataset_name"]
        model_name = self.config["model_name"]
        batch_size = self.config["batch_size"]

        self.log_path = os.path.join(log_dir, dataset_name, model_name)
        print("log path: ", self.log_path)
        if os.path.exists(self.log_path):
            print("A model already exists at this path %s\n. Refusing to overwrite "
            "previously trained model. Exiting program..." % self.log_path)
            exit(1)
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir)
        # Save the yaml configuration file
        os.system("cp %s %s" % (self.config_path, models_dir))

        # Initialize tensorboard writers
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.train_loader, self.val_loader, self.class_names = self.build_dataloaders()
        self.val_iter = iter(self.val_loader)
        print("Number of training examples: ", len(self.train_loader)*batch_size)
        print("Number of validation examples: ", len(self.val_loader)*batch_size)
        print("Number of steps per epoch: ", len(self.train_loader))

        weight_decay = self.config["weight_decay"]
        momentum = self.config["momentum"]
        lr = self.config["learning_rate"]
        scheduler_step_size = self.config["scheduler_step_size"]
        self.optim = torch.optim.SGD(self.model.parameters(), lr, momentum=momentum,
                                     weight_decay=weight_decay)
        # Use gamma as 0.1 as specified in paper
        # TODO: Do not know what scheduler step size the paper used...
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, scheduler_step_size, gamma=0.1)

        loss_dict = {
            "supervised": SupervisedLoss
        }

        # Define our loss function
        loss_type = self.config["loss_type"]
        self.loss_fn = loss_dict[loss_type](self.config)

        batch_size = self.config["batch_size"]
        num_epochs = self.config["num_epochs"]
        self.num_steps = len(self.train_loader) * num_epochs

        # Keep track of number of training iterations we take
        self.step = 0

        self.start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.run_epoch()
            self.lr_scheduler.step()
            if (epoch + 1) % self.config["save_frequency"] == 0:
                self.save_model()


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_mode("eval")
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
        input_image, y = self.process_inputs(inputs)

        with torch.no_grad():
            y_hat = self.model.forward(input_image)
            loss = self.loss_fn.forward(y_hat, y)
            self.log(input_image, y_hat, y, loss, mode="val")

        self.set_mode("train")

    def inference(self):
        self.set_mode("eval")
        train_loader, val_loader, class_names = self.build_dataloaders(use_all=True)
        train_iter, val_iter = iter(train_loader), iter(val_loader)
        inputs = self.process_inputs(train_iter.next())
        images, y = inputs[0], inputs[1]
        # [N, 1]
        y_hat_train = torch.argmax(self.model.forward(images), axis=1)
        # Total examples minus examples that are wrong
        diff = y_hat_train - y
        num_correct_train = diff[diff == 0].shape[0]
        percent_correct_train = (100 * num_correct_train) / y.shape[0]
        print("num_correct_train: ", num_correct_train)
        print("percent_correct_train: ", percent_correct_train)

        images, y = self.process_inputs(val_iter.next())
        y_hat_val = torch.argmax(self.model.forward(images), axis=1)
        diff = y_hat_val - y
        num_correct_val = diff[diff == 0].shape[0]
        percent_correct_val = (100 * num_correct_val) / y.shape[0]
        print("num_correct_val: ", num_correct_val)
        print("percent_correct_val: ", percent_correct_val)
         






if __name__ == "__main__":
    yaml_path = "configs/config.yaml"
    model = Model(yaml_path)
    model.train()