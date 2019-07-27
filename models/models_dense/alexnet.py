
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from IPython import embed

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2), # 1 --> 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, dilation=4), # 4 --> 8
            nn.Conv2d(64, 192, kernel_size=5, padding=16, dilation=8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, dilation=8), # 8 --> 16
            nn.Conv2d(192, 384, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, dilation=16) # 16 --> 32
            ])

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

