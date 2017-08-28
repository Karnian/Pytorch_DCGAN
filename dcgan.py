import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from network import *

parser = argparse.ArgumentParser(description='Easy Implementation of DCGAN')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=64)  # DB: CelebA or LSUN
parser.add_argument('--z_dim', type=int, default=100)

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# misc
parser.add_argument('--db', type=str, default='celeb')
parser.add_argument('--model_path', type=str, default='./models')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./results')  # Results
parser.add_argument('--image_path', type=str, default='./CelebA/128_crop')  # Training Image Directory
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=50)

##### Helper Function for Image Loading
class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        # os.listdir Function gives all lists of directory
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        # Transform
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

######################### Main Function
def main():
    # Pre-Settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    # Pre-processing
    transform = transforms.Compose([
        transforms.Scale((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.db == 'lsun':
        dataset = datasets.LSUN('.', classes=['bedroom_train'], transform=transform)
    else:
        dataset = ImageFolder(args.image_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    generator = Generator(args.z_dim)
    discriminator = Discriminator()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), args.lr, [args.beta1, args.beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), args.lr, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    """Train generator and discriminator."""
    fixed_noise = to_variable(torch.randn(args.batch_size, args.z_dim))  # For Testing
    total_step = len(data_loader)  # For Print Log
    for epoch in range(args.num_epochs):
        for i, images in enumerate(data_loader):
            # ===================== Train D =====================#
            if args.db == 'lsun':  # To Remove Label of LSUN DB
                images = to_variable(images[0])
            else:  # CelebA DB has no Label
                images = to_variable(images)

            batch_size = images.size(0)
            noise = to_variable(torch.randn(batch_size, args.z_dim))

            # Fake -> Fake & Real -> Real
            fake_images = generator(noise)
            d_loss = -torch.mean(torch.log(discriminator(images) + 1e-8) + torch.log(1 - discriminator(fake_images) + 1e-8))

            # Optimization
            discriminator.zero_grad()
            generator.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===================== Train G =====================#
            noise = to_variable(torch.randn(batch_size, args.z_dim))

            # Fake -> Real
            fake_images = generator(noise)
            g_loss = -torch.mean(torch.log(discriminator(fake_images) + 1e-8))

            # Optimization
            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, total_step, d_loss.data[0], g_loss.data[0]))
            # save the sampled images
            if (i + 1) % args.sample_step == 0:
                fake_images = generator(fixed_noise)
                torchvision.utils.save_image(denorm(fake_images.data),
                                             os.path.join(args.sample_path,
                                                          'generatedimage-%d-%d.png' % (epoch + 1, i + 1)))

        # save the model parameters for each epoch
        g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
        torch.save(generator.state_dict(), g_path)

if __name__ == "__main__":
    main()