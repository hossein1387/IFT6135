from model import Generator, Discriminator
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import utility

z_dim = 128
cuda_available = torch.cuda.is_available()


def build_model():
    generator = Generator()
    discriminator = Discriminator()
    if cuda_available:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    loss = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

    return generator, discriminator, loss, optimizer_g, optimizer_d

def train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d):
    ctr = 0
    minibatch_disc_losses = []
    minibatch_gen_losses = []

    for epoch in range(50):
        losses = []
        # Train
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            ctr += 1
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            zeros = Variable(torch.zeros(inputs.size(0)))
            ones = Variable(torch.ones(inputs.size(0)))

            if cuda_available:
                zeros, ones = zeros.cuda(), ones.cuda()

            ############################
            # (1) Update Discriminator
            ############################

            # Sample z ~ N(0, 1)
            minibatch_noise = Variable(torch.from_numpy(
                np.random.randn(inputs.size(0), z_dim, 1, 1).astype(np.float32)
            ))

            if cuda_available:
                minibatch_noise = minibatch_noise.cuda()

            # Zero gradients for the discriminator
            optimizer_d.zero_grad()

            # Train with real examples
            d_real = discriminator(inputs)

            d_real_loss = loss(d_real, ones)  # Train discriminator to recognize real examples
            d_real_loss.backward()

            # Train with fake examples from the generator
            fake = generator(minibatch_noise).detach()  # Detach to prevent backpropping through the generator
            d_fake = discriminator(fake)

            d_fake_loss = loss(d_fake, zeros)  # Train discriminator to recognize generator samples
            d_fake_loss.backward()
            minibatch_disc_losses.append(d_real_loss.data[0] + d_fake_loss.data[0])

            # Update the discriminator
            optimizer_d.step()

            ############################
            # (2) Update Generator
            ############################

            # Zero gradients for the generator
            optimizer_g.zero_grad()

            # Sample z ~ N(0, 1)
            minibatch_noise = Variable(torch.from_numpy(
                np.random.randn(inputs.size(0), z_dim, 1, 1).astype(np.float32)
            ))

            if cuda_available:
                minibatch_noise = minibatch_noise.cuda()

            d_fake = discriminator(generator(minibatch_noise))
            g_loss = loss(d_fake, ones)  # Train generator to fool the discriminator into thinking these are real.

            g_loss.backward()

            # Update the generator
            optimizer_g.step()

            minibatch_gen_losses.append(g_loss.data[0])

        print('Generator loss : %.3f' % (np.mean(minibatch_gen_losses)))
        print('Discriminator loss : %.3f' % (np.mean(minibatch_disc_losses)))

def eval(generator):
    # Set generator in evaluation mode to use running means and averages for Batchnorm
    generator.eval()

    # Sample z ~ N(0, 1)
    minibatch_noise = Variable(torch.from_numpy(
        np.random.randn(16, z_dim, 1, 1).astype(np.float32)
    ))

    if cuda_available:
        minibatch_noise = minibatch_noise.cuda()

    fakes = generator(minibatch_noise)

    fig = plt.figure(figsize=(10, 10))
    idx = 1
    for ind, fake in enumerate(fakes):
        fig.add_subplot(4, 4, ind + 1)
        plt.imshow(fake.data.cpu().numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')

def main():
    generator, discriminator, loss, optimizer_g, optimizer_d = build_model()
    trainloader = utility.trainloader()
    train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d)
    eval(generator)

if __name__ == '__main__':
    main()