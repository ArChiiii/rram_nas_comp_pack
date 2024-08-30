import torch
import torch.nn as nn

class MobileNetV3_Small(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000, dropout_prob = 0.2):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HSwish(),
        )

        self.bottlenecks = nn.Sequential(
            Bottleneck(input_channels=16, kernel=3, stride=1, expansion=16, output_channels=16, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=16, kernel=3, stride=2, expansion=72, output_channels=24, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=24, kernel=3, stride=1, expansion=88, output_channels=24, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=24, kernel=5, stride=2, expansion=96, output_channels=40, activation=HSwish()),
            Bottleneck(input_channels=40, kernel=5, stride=1, expansion=240, output_channels=40, activation=HSwish()),
            Bottleneck(input_channels=40, kernel=5, stride=1, expansion=240, output_channels=40, activation=HSwish()),
            Bottleneck(input_channels=40, kernel=5, stride=1, expansion=120, output_channels=48, activation=HSwish()),
            Bottleneck(input_channels=48, kernel=5, stride=1, expansion=144, output_channels=48, activation=HSwish()),
            Bottleneck(input_channels=48, kernel=5, stride=2, expansion=288, output_channels=96, activation=HSwish()),
            Bottleneck(input_channels=96, kernel=5, stride=1, expansion=576, output_channels=96, activation=HSwish()),
            Bottleneck(input_channels=96, kernel=5, stride=1, expansion=576, output_channels=96, activation=HSwish()),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(576),
            HSwish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(576, 1280),
            HSwish(),
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Linear(1280, num_classes),
            # you may add your own final-layer activation function here, based on your use case
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bottlenecks(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, input_channels, kernel, stride, expansion, output_channels, activation):
        super().__init__()

        self.bottleneck = nn.Sequential(
            # expansion
            nn.Conv2d(input_channels, expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expansion),
            activation,

            # depth-wise convolution
            nn.Conv2d(expansion, expansion, kernel_size=kernel, stride=stride, padding=kernel//2, groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            activation,

            # squeeze-and-excite
            SqueezeExcite(expansion),
            
            # point-wise convolution
            nn.Conv2d(expansion, output_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            activation,
        )
        
        # for residual skip connecting when the input size is different from output size
        self.downsample = None if input_channels == output_channels and stride == 1 else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
        )


    def forward(self, x):
        residual = x
        output = self.bottleneck(x)

        if self.downsample:
            residual = self.downsample(x)

        output = output + residual

        return output


class SqueezeExcite(nn.Module):
    def __init__(self, input_channels, squeeze = 4):
        super().__init__()

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(input_channels, out_channels=input_channels//squeeze, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(input_channels//squeeze),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//squeeze, input_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(input_channels),
            HSigmoid(),
        )
    
    def forward(self, x):
        x = x * self.SE(x)
        return x


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = x * self.relu6(x + 3) / 6
        return x


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.relu6(x + 3) / 6
        return x