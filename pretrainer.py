from torch.nn.modules.activation import LeakyReLU
from torch.serialization import check_module_version_greater_or_equal
from generator import G
from dataset import PRollDataset
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

BATCH_SIZE = 64

def padder(batch):
    max_dim = 0 
    for elem in batch:
        if(elem.shape[0] > max_dim):
            max_dim = elem.shape[0]
    
    for i in range(len(batch)):
        batch[i] = F.pad(batch[i], pad=(0, 0, 0, max_dim - batch[i].shape[0]))

    return torch.stack(batch, dim=0)


class G_pretrain(nn.Module):
    def __init__(self, noise_size, song_size):
        super().__init__()
        
        hidden_size_1 = 128
        lstm_input_size = 32
         
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(song_size, hidden_size_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size_1, lstm_input_size),
            nn.LeakyReLU()
        )

        self.encoder_lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = noise_size,
            num_layers = 2,
            batch_first = True,
            dropout = 0.0,
            bidirectional = False
        )

        self.decoder = G(noise_size, song_size)


    def forward(self, input):
        linear = self.encoder_linear(input)
        o, (h, _) = self.encoder_lstm(linear)
        output = self.decoder(o)
        return output


def custom_loss(inputs, targets):
    return torch.abs(torch.mean((inputs - targets))) * 10


if __name__ == "__main__":

    EPOCHS = 20
    BATCH_SIZE = 64
    NOISE_SIZE = 3
    CHECKPOINT_PATH = "pretraining_checkpoints/checkpoint_"

    g = G_pretrain(NOISE_SIZE, 276).to("cuda")
    opt = optim.Adam(params=g.parameters())
    criterion = nn.MSELoss()
    dataset = PRollDataset("dataset_preprocessed_reduced", test=False, device="cuda")

    train_dataset_length = int(len(dataset) * .7)
    eval_dataset_length = len(dataset) - train_dataset_length

    train_set, val_set = torch.utils.data.random_split(dataset, [train_dataset_length, eval_dataset_length])



    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=padder)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=padder)


    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        epoch_eval_loss = 0
        
        g.train()
        for batch in tqdm(train_loader, disable=True):
            opt.zero_grad()
            mask = torch.randperm(batch.shape[1])[:150]
            masked_batch = batch.clone()
            masked_batch[:, mask, :] = -1
            generated_song = g(masked_batch)
            loss = custom_loss(generated_song[:, mask, :], batch[:, mask, :])
            #loss = custom_loss(batch, generated_song)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item()

        g.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, disable=True):
                mask = torch.randperm(batch.shape[1])[:100]
                masked_batch = batch.clone()
                masked_batch[:, mask, :] = -1
                generated_song = g(masked_batch)
                #loss = criterion(generated_song[:, mask, :], batch[:, mask, :])
                loss = custom_loss(generated_song[:, mask, :], batch[:, mask, :])
                epoch_eval_loss += loss.item()

                
            

        
        torch.save({
                "epoch": epoch,
                "generator": g.decoder.state_dict(),
                "optimizer_g": opt.state_dict()
            }, CHECKPOINT_PATH + str(epoch+1) + ".pt")

        print(f"Train: {epoch_train_loss / len(train_loader)} \t Test: {epoch_eval_loss / len(val_loader)}")
            

