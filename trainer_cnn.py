from torch.utils.data import dataset
from cnn_generator import G
from cnn_discriminator import D
from dataset import PRollDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm.auto import tqdm
import numpy as np
import os
from torch import nn


def padder(batch):
    max_dim = 0 
    for elem in batch:
        if(elem.shape[0] > max_dim):
            max_dim = elem.shape[0]
    
    for i in range(len(batch)):
        batch[i] = F.pad(batch[i], pad=(0, 0, 0, max_dim - batch[i].shape[0]))
        #batch[i] = batch[i][:500, :]

    return torch.stack(batch, dim=0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(g, d, loader, ex_g_loss, ex_d_loss, opt_g, opt_d, epoch_n):
    
    # REMEMBER that the discriminator now has 1 output node only

    g.train()
    d.train()
    
    avg_err_D = 0
    avg_err_G = 0
    avg_D_real = 0
    avg_D_fake = 0
    n_batches = 0
    
    for i, batch in tqdm(enumerate(loader), desc="Epoch "+str(epoch_n)+": ", total=len(loader)):

    
        b_size = batch.shape[0]
       
        batch = torch.unsqueeze(batch, dim=1)
        
        # TRAIN D
        # Train with all real batch
        
        d.zero_grad()
        output = d(batch).view(-1)
        #label = torch.full((b_size, ), 1., dtype=torch.float, device=DEVICE)
        label = torch.FloatTensor(b_size).uniform_(0.8, 1.0).to(DEVICE)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        # Train with all fake batch
        
        noise = torch.randn(b_size, NOISE_SIZE, 1, 1, device=DEVICE)
        fake = g(noise)
        #label.fill_(0.)
        label = torch.FloatTensor(b_size).uniform_(0, 0.2).to(DEVICE)
        output = d(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        errD_fake.backward()
        
        opt_d.step()
        
        # TRAIN G
        
        g.zero_grad()
        #label.fill_(1.)
        label = torch.FloatTensor(b_size).uniform_(0.9, 1.0).to(DEVICE)
        output = d(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        opt_g.step()

        avg_err_D += errD
        avg_err_G += errG
        avg_D_real += D_x
        avg_D_fake += D_G_z1

        n_batches = i + 1

        
        
    avg_err_D /= n_batches
    avg_err_G /= n_batches
    avg_D_real /= n_batches
    avg_D_fake /= n_batches
    

    return avg_err_D, avg_err_G, avg_D_real, avg_D_fake



# params
BATCH_SIZE = 64
NOISE_SIZE = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TONES_NUMBER = 68 + 1
TEST = False
G_PRETRAIN = 3
LOAD_PRETRAINED_GENERATOR = False
EPOCHS = 1000
CHECKPOINT_PATH = "checkpoints/checkpoint_"
PRETRAINED_CHECKPOINT_PATH = "pretraining_checkpoints/checkpoint_"
WRITER_PATH = ""

torch.manual_seed(0)

dataset = PRollDataset("dataset_preprocessed_reduced", device="cuda", test=TEST)
criterion = BCELoss()

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=padder)

d = D(TONES_NUMBER*4).to(DEVICE)
g = G(NOISE_SIZE, TONES_NUMBER*4).to(DEVICE)
d.apply(weights_init)
g.apply(weights_init)
optimizer_d = optim.Adam(d.parameters(), betas=(0.5, 0.999))
optimizer_g = optim.Adam(g.parameters(), betas=(0.5, 0.999))

if(WRITER_PATH == ""):
    writer = SummaryWriter()
else:
    writer = SummaryWriter(WRITER_PATH)


checkpoints = os.listdir(CHECKPOINT_PATH.split("/")[0])
pretrained_checkpoints = []
if(LOAD_PRETRAINED_GENERATOR):
    pretrained_checkpoints = os.listdir(PRETRAINED_CHECKPOINT_PATH.split("/")[0])

if(LOAD_PRETRAINED_GENERATOR and len(pretrained_checkpoints) > 0 and len(checkpoints) == 0):
    last_checkpoint_path = PRETRAINED_CHECKPOINT_PATH+str(len(pretrained_checkpoints))+".pt"
    last_checkpoint = torch.load(last_checkpoint_path)
    g.load_state_dict(last_checkpoint["generator"])


last_epoch = 0
if(not TEST and len(checkpoints) > 0):
    last_checkpoint_path = CHECKPOINT_PATH+str(len(checkpoints))+".pt"
    last_checkpoint = torch.load(last_checkpoint_path)
    g.load_state_dict(last_checkpoint["generator"])
    d.load_state_dict(last_checkpoint["discriminator"])
    optimizer_g.load_state_dict(last_checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(last_checkpoint["optimizer_d"])
    last_epoch = last_checkpoint["epoch"]




avg_err_D = 1000
avg_err_G = 1000
for epoch in range(EPOCHS):
    epoch += last_epoch + 1
    avg_err_D, avg_err_G, avg_D_real, avg_D_fake = train(g, d, train_loader, avg_err_G, avg_err_D, optimizer_g, optimizer_d, epoch)
    print(f"D_loss: {avg_err_D}, G_loss: {avg_err_G}, D_real: {avg_D_real}, D_fake: {avg_D_fake}")

    # Save model
    if(not TEST):
        # Tensboard data
        writer.add_scalar("Loss/Discriminator", avg_err_D, epoch)
        writer.add_scalar("Loss/Generator", avg_err_G, epoch)
        writer.add_scalar("Discriminator/Real", avg_D_real, epoch)
        writer.add_scalar("Discriminator/Fake", avg_D_fake, epoch)
        torch.save({
            "epoch": epoch,
            "generator": g.state_dict(),
            "discriminator": d.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }, CHECKPOINT_PATH + str(epoch) + ".pt")


