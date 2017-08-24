import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # [-1, 100, 1, 1] -> [-1, 1024, 4, 4]
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0),

            # [-1, 512, 8, 8]
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # [-1, 256, 16, 16]
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # [-1, 128, 32, 32]
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # [-1, 64, 3, 3]
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        # [-1, 100] -> [-1, 100, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main( z )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # [-1, 3, 64, 64] -> [-1, 128, 32, 32]
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(0.05, inplace=True),

            # [-1, 256, 16, 16]
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 512, 8, 8]
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 1024, 4, 4]
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 1, 1, 1]
            nn.Conv2d(1024, 1, 4, 1, 0)
        )
    def forward(self, x):
        # [-1, 1, 1, 1] -> [-1, 1]
        return F.sigmoid(self.main(x).squeeze())
