import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import utils

class Generator(nn.Module):
    def __init__(self, inp_dim=62):
        super(Generator, self).__init__()
        self.inp_dim = inp_dim
        self.inp_height = 28
        self.inp_width = 28
        self.n_color_channels = 1

        self.fc = nn.Sequential(
                nn.Linear(self.inp_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * (self.inp_width // 4) * (self.inp_height // 4)),
                nn.BatchNorm1d(128 * (self.inp_width // 4) * (self.inp_height // 4)),
                nn.ReLU()
        )
        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.n_color_channels, 4, 2, 1),
                nn.Sigmoid()        # all values must be between [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        # change into 3d minibatch form for deconvolution
        x = x.view(-1, 128, (self.inp_width // 4), (self.inp_height // 4))
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.inp_width = 28
        self.inp_height = 28
        self.n_color_channels = 1

        self.conv = nn.Sequential(
                nn.Conv2d(self.n_color_channels, 64, 4, 2, 1),  # reduce size to 1/2 of original
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.6),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.6)
        )

        self.fc = nn.Sequential(
                nn.Linear(128 * (self.inp_width // 4) * (self.inp_height // 4), 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        # change into 1d minibatch form for fc
        x = x.view(-1, 128 * (self.inp_width // 4) * (self.inp_height // 4))
        x = self.fc(x)
        return x

class GAN:
    def __init__(self, args, data_loader):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.inp_dim = args.inp_dim
        self.data_loader = data_loader

        self.G = Generator(self.inp_dim)
        self.D = Discriminator()

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lr_G)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lr_D)

        self.criterion = nn.BCELoss()

        self.sample_noise = Variable(torch.rand((self.batch_size, self.inp_dim)), volatile=True)

    def train(self):
        self.train_hist = dict()
        self.train_hist["D_loss"] = list()
        self.train_hist["G_loss"] = list()

        y_real = Variable(torch.ones((self.batch_size, 1)))
        y_fake = Variable(torch.zeros((self.batch_size, 1)))

        for epoch in range(self.epoch):
            self.G.train()
            for step, (x_real, _) in enumerate(self.data_loader):
                if step == self.data_loader.__len__() - 1:
                    # ignore the last batch
                    break

                z = Variable(torch.rand(self.batch_size, self.inp_dim))
                x_real = Variable(x_real)

                # train Discriminator
                self.D_optimizer.zero_grad()

                # loss when D is identifying real dataset(should predict 1)
                D_real = self.D(x_real)
                D_real_loss = self.criterion(D_real, y_real)

                # loss when D is identifying fake dataset(should predict 0)
                x_fake = self.G(z)
                D_fake = self.D(x_fake)
                D_fake_loss = self.criterion(D_fake, y_fake)

                D_loss = D_fake_loss + D_real_loss
                self.train_hist["D_loss"].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # train Generator
                self.G_optimizer.zero_grad()
                x_generated = self.G(z)
                D_fake = self.D(x_generated)

                # Generator should aim to make prediction of Discriminator
                # close to 1
                G_loss = self.criterion(D_fake, y_real)
                self.train_hist["G_loss"].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if (step + 1) % 100 == 0:
                    print("Epoch: [%2d] D_loss: %.5f G_loss: %.5f" % (epoch + 1, D_loss.data[0], G_loss.data[0]))

            self.visualize(epoch + 1)


        print("Training complete!!")
        utils.plot_losses(self.train_hist)

    def visualize(self, epoch):
        self.G.eval()

        # batch_size should be greater or equal to than 16
        frame_size = 4
        samples = self.G(self.sample_noise).squeeze().data.numpy().transpose(0, 1, 2)
        samples = samples[:frame_size * frame_size]
        fig, a = plt.subplots(frame_size, frame_size, figsize=(8, 8))

        curr = 0
        for i in range(frame_size):
            for j in range(frame_size):
                a[i][j].clear()
                a[i][j].imshow(samples[curr], cmap='gray')
                a[i][j].set_xticks(())
                a[i][j].set_yticks(())
                curr += 1
        plt.savefig("./figures/GAN_" + str(epoch) + ".png")
