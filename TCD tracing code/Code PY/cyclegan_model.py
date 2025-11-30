import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3),
            nn.InstanceNorm1d(dim),
            nn.ReLU(True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3),
            nn.InstanceNorm1d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, n_residual_blocks=6):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm1d(64),
            nn.ReLU(True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv1d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResNetBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose1d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm1d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(64, output_channels, kernel_size=7),
            nn.Tanh() # Output range [-1, 1], assuming input is normalized similarly
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad1d((1, 0)),
            nn.Conv1d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
