from numpy.core.numeric import outer
from torch.serialization import load
from torch.utils.data import dataloader, dataset
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


def padder(batch):
    max_dim = 0 
    for elem in batch:
        if(elem.shape[0] > max_dim):
            max_dim = elem.shape[0]
    
    for i in range(len(batch)):
        batch[i] = F.pad(batch[i], pad=(0, 0, 0, max_dim - batch[i].shape[0]))
        #batch[i] = batch[i][:500, :]

    return torch.stack(batch, dim=0)


def train(g, d, loader, g_loss_function, d_loss_function, opt_g, opt_d):
    
    g.train()
    d.train()
    
    avg_loss_g = 0
    avg_loss_d = 0
    
    for i, batch in tqdm(enumerate(loader), desc="Training"):

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
BATCH_SIZE = 32
NOISE_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TONES_NUMBER = 68 + 1
torch.manual_seed(0)

dataset = PRollDataset("dataset_preprocessed", device="cuda", test=True)

train_length = int(len(dataset) * 0.75)
test_length = int(len(dataset) * 0.10)
evaluation_length = len(dataset) - train_length - test_length
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length, evaluation_length))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padder)
dev_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

d = D(TONES_NUMBER*4).to(DEVICE)
g = G(NOISE_SIZE, TONES_NUMBER*4).to(DEVICE)
optimizer_d = optim.Adam(params=d.parameters())
optimizer_g = optim.Adam(params=g.parameters())


writer = SummaryWriter()
EPOCHS = 1000
CHECKPOINT_PATH = "checkpoints/checkpoint_"

for epoch in range(EPOCHS):
    avg_loss_d, avg_loss_g = train(g, d, train_loader, l_g, l_d, optimizer_g, optimizer_d)
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



