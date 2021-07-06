import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def resnet(num_layers, num_classes, pretrained=False):
    """
    Link to pytorch resnet implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Args:
        num_layers (int): number of layers for resnet
        num_classes (int): number of output classes
        pretrained (bool): Whether or not to use pretrained imagenet weights

    Returns:
        The ResNet model (torchvision.models.resnet.ResNet)    
    """

    block_dict = {
        18: models.resnet.BasicBlock,
        34: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck,
        101: models.resnet.Bottleneck,
        152: models.resnet.Bottleneck
    }
    layers_dict = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    block = block_dict[num_layers]
    layers = layers_dict[num_layers]

    model = models.resnet.ResNet(layers=layers, block=block, num_classes=num_classes)

    if pretrained:
        string = 'resnet%d' % num_layers
        pretrained_dict = model_zoo.load_url(
            models.resnet.model_urls[string]
        )

        # Only load pretrained image-net weights for layers which are not final fc layer
        # as final fc layer has shape mis-match
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key in pretrained_dict.keys():
                model_dict[key] = pretrained_dict[key]
        model.load_state_dict(model_dict)

    return model


if __name__ == "__main__":
    model = resnet(num_layers=18, num_classes=10, pretrained=True)
    print(model)

