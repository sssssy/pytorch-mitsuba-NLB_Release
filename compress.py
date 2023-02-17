'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    	
    Jiahui Fan, 2022/09
'''

'''
    This script compress a given BRDF data file (.npy) into latent code (32 x 3).
'''

import random, sys

import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

import model
from utils import *

GLOBAL_RANDOM_SEED = datetime.now().microsecond
torch.manual_seed(GLOBAL_RANDOM_SEED)
torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)
torch.backends.cudnn.deterministic = True
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
torch.set_printoptions(sci_mode=False, linewidth=110)
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print(
    '''
        Usage: $ python compress.py NPY_PATH
        
        Args:
            NPY_PATH: the path to BRDF data file 
    '''
        )
        exit()

    file_data = torch.tensor(np.load(sys.argv[1]).reshape(1, -1, 7)).float()
    print(file_data.shape)

    config = DecoderOnlyConfig()
    decoder = getattr(model, config.compress_decoder)(config)
    decoder.load_state_dict(torch.load(config.decoder_path)())
    decoder = decoder.cuda().train()

    batch_size = 4096
    max_steps = max(int(file_data.shape[1] // batch_size) * 10, 1000) ## empirically, at least 1k steps or 10 epochs
    lr = 0.001
    train_lr = lr * 0.003
    lr_decay_freq = max_steps // 5
    lr_decay = (train_lr / lr) ** (1.2 / max_steps * lr_decay_freq)
    criterion = nn.L1Loss()
    resolution = 512
    latent_size = config.latent_size

    latent = torch.ones([1, latent_size]).cuda()
    latent.requires_grad = True
    avg_loss = []

    optimizer = torch.optim.Adam(
        [{'params': latent, 'lr': lr}]
    )

    reset_perm = file_data.shape[-2] // batch_size
    batch_num = 0
    while True:
        if batch_num % reset_perm == 0:
            random_perm = torch.randperm(file_data.shape[-2])
        random_index = random_perm[batch_num % reset_perm: batch_num%reset_perm + batch_size]
        wiwo = file_data[:, random_index, :4].cuda()
        wiwo = torch.stack([wiwo, wiwo, wiwo], dim=2).reshape(file_data.shape[0], -1, 4) #!
        rgb = file_data[:, random_index, -3:].cuda()
        wiwo = to6d(wiwo)

        cur_file_latent = latent[0].reshape(1, 1, -1)
        latent_expand = cur_file_latent.expand(wiwo.shape[0], batch_size, -1).reshape(
            wiwo.shape[0], -1, config.latent_size // 3
        ) #!
        batch_input = torch.cat([wiwo, latent_expand], axis=-1)
        output = decoder(batch_input) * wiwo[:, :, -1:]
        output = output.reshape(file_data.shape[0], -1, 3) #!

        loss = criterion(output / (1 + output), rgb / (1 + rgb))
        avg_loss.append(torch.abs(output - rgb).mean().item())
        if batch_num % lr_decay_freq == 0:
            line = '>>> step {}, avg_loss: {:.4f} +- ({:.4f}), latent.mean: {:.4f}, lr: {:.7f}'.format(
                batch_num,
                np.mean(avg_loss), np.std(avg_loss),
                cur_file_latent.mean(), optimizer.param_groups[0]['lr']
            )
            print('\r'+line, end='')
            avg_loss = []
            if batch_num >= max_steps:
                print('\r'+line, end='')
                break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_num += 1
        if batch_num % lr_decay_freq == 0 and batch_num <= 0.6 * max_steps:
            optimizer.param_groups[0]['lr'] *= lr_decay

    line = '{}'.format(cur_file_latent)
    print('\n'+line)