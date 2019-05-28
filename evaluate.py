# coding: utf-8
# Import SGAN models
from model import *

# use idel gpu
# it's better to use environment variable
# if you want to use multiple GPUs, please
# modify hyperparameters at the same time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1, 2'
n_gpu             = 2
device            = torch.device('cuda:0')

# Hyper-parameters
n_fc              = 8
dim_latent        = 512
dim_input         = 4
step              = 7
resolution        = 2 ** (step + 2)
save_folder_path  = './results/'

# Style mixing setting
style_mixing      = []
low_steps         = [0, 1, 2]
# style_mixing    += low_steps
mid_steps         = [3, 4, 5]
# style_mixing    += mid_steps
hig_steps         = [6, 7, 8]
# style_mixing    += hig_steps

generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
if os.path.exists('checkpoint/trained.pth'):
    checkpoint = torch.load('checkpoint/trained.pth')
    generator.load_state_dict(checkpoint['generator'])
else:
    raise IOError('No checkpoint file found at ./checkpoint/trained.pth')
generator.eval()
# No computing gradients
for param in generator.parameters():
    param.requires_grad = False

def compute_latent_cernter(batch_size=1024, multimes=10):
    appro_latent_center = None
    for i in range(multimes):
        if appro_latent_center is None:
            appro_latent_center = generator.center_w(torch.randn((batch_size, dim_latent)).to(device))
        else:
            appro_latent_center += generator.center_w(torch.randn((batch_size, dim_latent)).to(device))
    appro_latent_center /= multimes
    return appro_latent_center

def evaluate(latent_code, noise, latent_w_center=None, psi=0, style_mixing=[]):
    if n_gpu > 1:
        return nn.parallel.data_parallel(generator, (latent_code, step, 1, noise, style_mixing,
            latent_w_center, psi), range(n_gpu))
    else:
        return generator(latent_code, step, 1, noise, style_mixing, latent_w_center, psi)