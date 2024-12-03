import cv2
from models import Generator,Discriminator,vggL
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import vgg19
import torch
from torch import optim
from tqdm import tqdm
import torchvision.transforms as transforms
from data_preprocess import ImageDataset
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
epochs = 5
batch_size = 16
num_workers = 0
img_channels = 3

gen = Generator(in_channels=3).to(device)
disc = Discriminator(in_channels=3).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = vggL()


train = ImageDataset(root_dir="dataset/train")
train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers)

val = ImageDataset(root_dir="dataset/val")
val_loader = DataLoader(val, batch_size=batch_size, num_workers=num_workers)


def plot_examples(gen,index):
    dataset_test = ImageDataset(root_dir="dataset/val")
    loader = DataLoader(dataset_test, batch_size=16, num_workers=0)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    chosen_batch = random.randint(0, len(loader) - 1)
    for idx, (low_res, high_res) in enumerate(loader):
        if (chosen_batch == idx):
            chosen = random.randint(0, len(low_res) - 1)

            # axs[0].set_axis_off()
            # axs[0].imshow(low_res[chosen].permute(1, 2, 0))
            # axs[0].set_title("low res")

            with torch.no_grad():
                upscaled_img = gen(low_res[chosen].to(device).unsqueeze(0))
                fake_img = upscaled_img.cpu().permute(0, 2, 3, 1)[0]
                fake_img = fake_img.detach().cpu().numpy()
                cv2.imwrite(f'images/pred_image{index}.jpg', fake_img * 255)
                real_img = high_res[chosen].permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite(f'images/real_image{index}.jpg',real_img*255)
    #             img = img[0].detach().cpu().numpy()
    #             cv2.imwrite(f'images/pred_image{index}.jpg',img)
    #
    #         axs[1].set_axis_off()
    #         axs[1].imshow(upscaled_img.cpu().permute(0, 2, 3, 1)[0])
    #         axs[1].set_title("predicted")
    #
    #         axs[2].set_axis_off()
    #         axs[2].imshow(high_res[chosen].permute(1, 2, 0))
    #         axs[2].set_title("high res")
    #         if (idx == 1):
    #             break
    #
    # # Show the figure
    # plt.show()
    gen.train()

def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader)
    disc_loss = 0
    gen_loss = 0

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(device)
        low_res = low_res.to(device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)

        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    return gen_loss.detach().cpu(), disc_loss.detach().cpu()

d_losses = []
g_losses = []
for epoch in range(epochs):
     plot_examples(gen,epoch)
     print("epoch ", epoch+1, "/", epochs)
     gen_loss, disc_loss = train_fn(train_loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)
     # train discriminator and generator and update losses
     d_losses.append(disc_loss)
     g_losses.append(gen_loss)



torch.save(gen.state_dict(), "checkpoint1_gen")
torch.save(disc.state_dict(), "checkpoint1_disc")

