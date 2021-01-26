from torch.utils.data import dataset
from generator import G
from discriminator import D
from losses import l_g, l_d
from dataset import PRollDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm.auto import tqdm
import numpy as np
import os


def padder(batch):
    max_dim = 0 
    for elem in batch:
        if(elem.shape[0] > max_dim):
            max_dim = elem.shape[0]
    
    for i in range(len(batch)):
        batch[i] = F.pad(batch[i], pad=(0, 0, 0, max_dim - batch[i].shape[0]))
        #batch[i] = batch[i][:500, :]

    return torch.stack(batch, dim=0)


def train(g, d, loader, g_loss_function, d_loss_function, opt_g, opt_d, epoch_n):
    
    g.train()
    d.train()
    
    avg_loss_g = 0
    avg_loss_d = 0
    
    for i, batch in tqdm(enumerate(loader), desc="Epoch " + str(epoch_n), total=len(loader)):

        # discriminator
        opt_d.zero_grad()
        
        d_real = d(batch)
        
        noise = torch.randn((batch.shape[0], batch.shape[1], NOISE_SIZE)).to(DEVICE)
        g_z = g(noise)


        
        d_fake = d(g_z.detach())

        # Remember: [:,0] is the true probability, [:,1] is (1 - [:,0])
        loss_d = d_loss_function(d_real[:, 0], d_fake[:, 1])
        loss_d.backward()
        opt_d.step()

        avg_loss_d += (loss_d.item() - avg_loss_d) / (i+1)
        
        # generator
        opt_g.zero_grad()
        opt_d.zero_grad()

        d_fake = d(g_z)
        loss_g = g_loss_function(d_fake[:, 1])
        loss_g.backward()
        opt_g.step()

        avg_loss_g += (loss_g.item() - avg_loss_g) / (i+1)

    return avg_loss_d, avg_loss_g



# params
BATCH_SIZE = 64
NOISE_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TONES_NUMBER = 68 + 1
torch.manual_seed(0)

dataset = PRollDataset("dataset_preprocessed_reduced", device="cuda", test=False)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=padder)

d = D(TONES_NUMBER*4).to(DEVICE)
g = G(NOISE_SIZE, TONES_NUMBER*4).to(DEVICE)
optimizer_d = optim.Adam(params=d.parameters())
optimizer_g = optim.Adam(params=g.parameters())

writer = SummaryWriter("runs/test")
EPOCHS = 1000
CHECKPOINT_PATH = "checkpoints/checkpoint_"

checkpoints = os.listdir("checkpoints")
last_epoch = 0
if(len(checkpoints) > 0):
    last_checkpoint_path = "checkpoints/checkpoint_"+str(len(checkpoints)-1)+".pt"
    last_checkpoint = torch.load(last_checkpoint_path)
    g.load_state_dict(last_checkpoint["generator"])
    d.load_state_dict(last_checkpoint["discriminator"])
    optimizer_g.load_state_dict(last_checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(last_checkpoint["optimizer_d"])
    last_epoch = last_checkpoint["epoch"]


    


for epoch in range(EPOCHS):
    epoch += last_epoch +1
    avg_loss_d, avg_loss_g = train(g, d, train_loader, l_g, l_d, optimizer_g, optimizer_d, epoch)
    # Tensboard data
    writer.add_scalar("Loss/Discriminator", avg_loss_d, epoch)
    writer.add_scalar("Loss/Generator", avg_loss_g, epoch)
    # Save model
    torch.save({
        "epoch": epoch,
        "generator": g.state_dict(),
        "discriminator": d.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }, CHECKPOINT_PATH + str(epoch) + ".pt")



g.eval()
with torch.no_grad():
    output = g(torch.randn((1, 500, NOISE_SIZE)).to(DEVICE)).cpu().numpy()
    np.save("output_test.npy", output)



