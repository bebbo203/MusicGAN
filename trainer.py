from torch.utils.data import dataset
from generator import G
from discriminator import D
from losses import l_g, l_d
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
        song_length = batch.shape[1]
        
        
        # TRAIN D
        # Train with all real batch
        d.zero_grad()
        output = d(batch).view(-1)
        label = torch.full((b_size, ), 1., dtype=torch.float, device=DEVICE)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        # Train with all fake batch
        noise = torch.randn(b_size, song_length, NOISE_SIZE, device=DEVICE)
        fake = g(noise)
        label.fill_(0.)
        output = d(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        if(epoch_n % G_PRETRAIN == 0):
            optimizer_d.step()

        # TRAIN G
        g.zero_grad()
        label.fill_(1.)
        output = d(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        if(epoch_n % G_PRETRAIN != 0):
            optimizer_g.step()

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
NOISE_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TONES_NUMBER = 68 + 1
TEST = False
G_PRETRAIN = 5
torch.manual_seed(0)

dataset = PRollDataset("dataset_preprocessed_reduced", device="cuda", test=TEST)
criterion = BCELoss()

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=padder)

d = D(TONES_NUMBER*4).to(DEVICE)
g = G(NOISE_SIZE, TONES_NUMBER*4).to(DEVICE)
optimizer_d = optim.Adam(params=d.parameters())
optimizer_g = optim.Adam(params=g.parameters())

writer = SummaryWriter()
EPOCHS = 1000
CHECKPOINT_PATH = "checkpoints/checkpoint_"

checkpoints = os.listdir("checkpoints")
last_epoch = 0
if(not TEST and len(checkpoints) > 0):
    last_checkpoint_path = "checkpoints/checkpoint_"+str(len(checkpoints)-1)+".pt"
    last_checkpoint = torch.load(last_checkpoint_path)
    g.load_state_dict(last_checkpoint["generator"])
    d.load_state_dict(last_checkpoint["discriminator"])
    optimizer_g.load_state_dict(last_checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(last_checkpoint["optimizer_d"])
    last_epoch = last_checkpoint["epoch"]


    


for epoch in range(EPOCHS):
    epoch += last_epoch + 1
    avg_err_D, avg_err_G, avg_D_real, avg_D_fake = train(g, d, train_loader, l_g, l_d, optimizer_g, optimizer_d, epoch)
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



g.eval()
with torch.no_grad():
    output = g(torch.randn((1, 500, NOISE_SIZE)).to(DEVICE)).cpu().numpy()
    np.save("output_test.npy", output)



