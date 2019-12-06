import torch.nn as nn
import torch.nn.functional as F


class C3D(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D, self).__init__()
        self.features = nn.Sequential()

        self.features.add_module(
            'conv1', nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'pool1', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.features.add_module(
            'conv2',
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'pool2', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.features.add_module(
            'conv3a',
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'conv3b',
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'pool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.features.add_module(
            'conv4a',
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'conv4b',
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'pool4', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.features.add_module(
            'conv5a',
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module(
            'conv5b',
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        out = self.features.forward(x)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)
        out = self.classifier(out)

        return out
