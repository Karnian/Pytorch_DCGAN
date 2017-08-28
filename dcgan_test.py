import argparse
import os
import torch
import torchvision
from torch.backends import cudnn
from torch.autograd import Variable
from network import Generator

parser = argparse.ArgumentParser(description='Easy Implementation of DCGAN')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=100) # Must be same as Train

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=5) # To Obtain N Epoch's Results
parser.add_argument('--sample_size', type=int, default=128) # Test Batch Size

# misc
parser.add_argument('--model_path', type=str, default='./models')  # Loading Model Path
parser.add_argument('--sample_path', type=str, default='./test_results')  # Save Result Path

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def main():
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (args.num_epochs))

    generator = Generator()
    generator.load_state_dict(torch.load(g_path))
    generator.eval()

    if torch.cuda.is_available():
        generator = generator.cuda()

    # Sample the images
    noise = to_variable(torch.randn(args.sample_size, args.z_dim))
    fake_images = generator(noise)
    sample_path = os.path.join(args.sample_path, 'fake_samples-final.png')
    torchvision.utils.save_image(denorm(fake_images.data), sample_path)

    print("Saved sampled images to '%s'" % sample_path)

if __name__ == "__main__":
    main()