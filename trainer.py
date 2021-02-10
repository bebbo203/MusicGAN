from torch.utils.data import dataset
from configuration import *
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from generator import G
from discriminator import D
from dataset import PRollDataset
from tester import generated_song_to_img
from PIL import Image
import os
import numpy as np


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    # Thanks to https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/5
    # Random interpolates between fake and real samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    # Compute gradients wrt interpolation
    fake = torch.ones(real_samples.size(0), 1).to(DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # <--- because gradients return a tuple of 1 object, i.e. the gradient that we want
    
    # Reshape to [batch_size, ...] shape
    gradients = gradients.view(gradients.size(0), -1)
    # Compute the euclidean norm
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def train(g, d, loader, opt_g, opt_d, epoch_n):

    g.train()
    d.train()
    
    avg_loss_D = 0
    avg_loss_G = 0
    avg_D_real = 0
    avg_D_fake = 0
    
    for i, batch in tqdm(enumerate(loader), desc="Epoch "+str(epoch_n)+": ", total=(len(loader) // N_SAMPLES_PER_SONG + 1)):

    
        b_size = batch.shape[0]
        z = torch.randn(b_size, LATENT_DIM).to(DEVICE)
       
        # TRAIN D
        # Train with all real batch
        
        d.zero_grad()
        D_x = d(batch)
        
        loss_D_x = -torch.mean(D_x) # maximize the outputs of real samples
        loss_D_x.backward()
        
        # Train with all fake batch
        G_z = g(z)
        D_G_z = d(G_z.detach())
        
        loss_D_G_z = torch.mean(D_G_z) # minimize the outputs of fake samples
        
        loss_D_G_z.backward()
        loss_D = loss_D_x + loss_D_G_z
        

        # Compute gradient penalty
        gradient_penalty = GAMMA * compute_gradient_penalty(d, batch.data, G_z.data)
        # Backpropagate the gradients
        gradient_penalty.backward()
        
        opt_d.step()

        # TRAIN G
        
        g.zero_grad()
        if(epoch_n % GENERATOR_UPDATE_INTERVAL == 0):    
            D_G_z = d(G_z)
            loss_G = -torch.mean(D_G_z) # maximize the outputs of fake samples
            loss_G.backward()
            opt_g.step()
        else:
            g.eval()
            with torch.no_grad():
                D_G_z = d(G_z)
                loss_G = -torch.mean(D_G_z)

        
        
    avg_loss_D /= (i + 1)
    avg_loss_G /= (i + 1)
    avg_D_real /= (i + 1)
    avg_D_fake /= (i + 1)
    

    return avg_loss_D, avg_loss_G, avg_D_real, avg_D_fake





torch.manual_seed(0)
g = G().to(DEVICE)
d = D().to(DEVICE)
optimizer_d = torch.optim.RMSprop(params=d.parameters(), lr=0.00005)
optimizer_g = torch.optim.RMSprop(params=g.parameters(), lr=0.00005)
dataset = PRollDataset("data", device=DEVICE, test=TEST)
collate = lambda batch: torch.cat([torch.tensor(b, dtype=torch.float32, device=DEVICE) for b in batch])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)


if(WRITER_PATH == ""):
    writer = SummaryWriter()
else:
    writer = SummaryWriter(WRITER_PATH)

checkpoints_path = CHECKPOINT_PATH.split("/")[:-1]
if(len(checkpoints_path) > 1):
    checkpoints_path = os.path.join(*checkpoints_path)
else:
    checkpoints_path = checkpoints_path[0]

checkpoints = os.listdir(checkpoints_path)      


last_epoch = 0
if(not TEST and len(checkpoints) > 0):
    last_checkpoint_path = CHECKPOINT_PATH+str(len(checkpoints))+".pt"
    last_checkpoint = torch.load(last_checkpoint_path)
    g.load_state_dict(last_checkpoint["generator"])
    d.load_state_dict(last_checkpoint["discriminator"])
    optimizer_g.load_state_dict(last_checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(last_checkpoint["optimizer_d"])
    last_epoch = last_checkpoint["epoch"]


noise = torch.randn(4, LATENT_DIM).to(DEVICE)

for epoch in range(EPOCHS):
    epoch += last_epoch + 1
    avg_loss_D, avg_loss_G, avg_D_real, avg_D_fake = train(g, d, train_loader, optimizer_g, optimizer_d, epoch)
    print(f"D_loss: {avg_loss_D:.5f}, G_loss: {avg_loss_G:.5f}, D_real: {avg_D_real:.5f}, D_fake: {avg_D_fake:.5f}")

    # Save model
    if(not TEST):
        # Tensorboard data
        writer.add_scalar("Loss/Discriminator", avg_loss_D, epoch)
        writer.add_scalar("Loss/Generator", avg_loss_G, epoch)
        writer.add_scalar("Discriminator/Real", avg_D_real, epoch)
        writer.add_scalar("Discriminator/Fake", avg_D_fake, epoch)
        torch.save({
            "epoch": epoch,
            "generator": g.state_dict(),
            "discriminator": d.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }, CHECKPOINT_PATH + str(epoch) + ".pt")

        if(epoch % LOG_INTERVAL == 0):
            with torch.no_grad():
                g.eval()
                generated_song = g(noise).cpu().numpy()
                img = generated_song_to_img(generated_song)
                writer.add_image("Generated Song", img, epoch)
                img = Image.fromarray(np.rollaxis(img, 0, 3))
                img.save(IMG_SAVE_PATH + "/epoch_" + str(epoch) + ".png")
