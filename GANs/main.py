import argparse
from GAN import GAN
from LSGAN import LSGAN
import torch.utils.data as Data
from torchvision import transforms, datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_type", type=str, default="GAN")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--lr_G", type=float, default=0.0002)
    parser.add_argument("--lr_D", type=float, default=0.0002)
    parser.add_argument("--inp_dim", type=int, default=62)

    return parser.parse_args()

def main():
    args = parse_args()
    data_loader = Data.DataLoader(datasets.FashionMNIST(
            root='data/fashion-MNIST',
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ), shuffle=True, batch_size=args.batch_size)
    if args.gan_type == "GAN":
        gan = GAN(args, data_loader)
    elif args.gan_type == "LSGAN":
        gan = LSGAN(args, data_loader)

    gan.train()

if __name__ == "__main__":
    main()
