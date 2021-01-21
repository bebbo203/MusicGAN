from generator import G
from discriminator import D
from dataset import MyDataset
import torch

def l_g(d_g_false):
    return torch.mean(torch.log(d_g_false))

def l_d(d_true, d_g_false):
    l1 = -torch.log(d_true)
    l2 = -l_g(d_g_false)
    return torch.mean(l1 + l2)



dataset = MyDataset("dataset", test=True)
song = torch.stack((dataset[0][:500, :], dataset[1][:500, :]), dim=0)

g = G(32, 128*4).to("cuda")
d = D(128*4).to("cuda")

noise = torch.randn((2, 96, 32)).to("cuda")
generated_songs = g(noise)
discriminated_songs_gen = d(generated_songs)
real_song_discriminated = d(song)

discriminated_songs_gen = torch.softmax(discriminated_songs_gen, dim=1)
discriminated_songs_gen_false = discriminated_songs_gen[:, 1]
d_true = torch.softmax(d(song), dim=1)

print(l_d(d_true, discriminated_songs_gen_false))

