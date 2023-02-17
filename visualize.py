'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    	
    Jiahui Fan, 2022/09
'''

'''
    This script helps visualize any latent code into outgoing radiance distributions.
'''

import sys

import torch
import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
import numpy as np

import exr
from utils import *
import model

'''
    Outgoing radiance visualization, given an representation network and a latent code.

    Args:
        decoder: torch.nn.Module, the network to visualize latent codes
        latent: torch.tensor, latent code to visualize of size (latent_size,)
        out_anme: save file name (.exr)
        resolution (optional): the resolution of output image (Default: 512)
        wiz (optional): z-coordinate of the view direction (Default: 1.0, i.e., [0, 0, 1] view direction)
'''
def vis_ndf(decoder, latent, out_name, resolution=512, wiz=1):
        
    wix = np.sqrt(1 - wiz**2)
    wiy = 0
    with torch.no_grad():
        new_wiwo = []
        reso_step = 2.0 / resolution
        for wox in np.arange(-1.0, 1.0, reso_step):
            for woy in np.arange(-1.0, 1.0, reso_step):
                new_wiwo.append([wix, wiy, wox, woy])
        new_wiwo = torch.Tensor(np.array(new_wiwo)).unsqueeze(0)  # size: [1, resolution ** 2, 4]
        new_wiwo = new_wiwo.cuda()
        new_wiwo = to6d(new_wiwo)

        latent = latent.reshape(1, 1, 32*3).expand(
            1, resolution**2, 32*3)
        decoder_input = torch.cat([new_wiwo, latent[:, :, :32]], axis=-1) # size: [1, reso**2, 20]
        output0 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        decoder_input = torch.cat([new_wiwo, latent[:, :, 32:-32]], axis=-1) # size: [1, reso**2, 20]
        output1 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        decoder_input = torch.cat([new_wiwo, latent[:, :, -32:]], axis=-1) # size: [1, reso**2, 20]
        output2 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        
        image = np.concatenate([output0, output1, output2], axis=0).reshape(3, resolution, resolution).transpose(1, 2, 0)

        ## drop out invalid points
        mid = resolution // 2
        for i in range(resolution):
            for j in range(resolution):
                distance = ((i - mid) / mid) ** 2 + ((j - mid) / mid) ** 2
                if distance > 1:
                    image[i, j, :] = 0.5

        exr.write32(image, out_name)
        
        
if __name__ == '__main__':

    if len(sys.argv) > 3:
        print(
        '''
            Usage: $ python visualize.py [resolution] [wiz]
            
            Args:
                resolution (optional): the resolution of output image (Default: 512)
                wiz (optional): z-coordinate of the view direction (Default: 1.0, i.e., [0, 0, 1] view direction)
        '''
            )
        exit()
        
    config = DecoderOnlyConfig()
    decoder = getattr(model, config.compress_decoder)(config)
    decoder.load_state_dict(torch.load(config.decoder_path)())
    decoder = decoder.cuda()

    latent = torch.tensor([

1.0645, 1.0681, 1.0385, 1.0532, 0.9485, 1.0549, 0.9796, 1.0457, 0.9864, 0.9645, 1.0280, 1.0639,
        1.0368, 0.9571, 0.9684, 0.9327, 1.0147, 0.9547, 0.9507, 0.9476, 0.9760, 1.0416, 0.9489, 1.0159,
        0.9936, 0.9458, 1.0620, 1.0448, 0.9574, 0.9258, 1.0414, 1.0986, 1.0487, 1.0598, 1.0448, 1.0532,
        0.9485, 1.0438, 0.9675, 1.0342, 1.0013, 0.9733, 1.0339, 1.0986, 1.0441, 0.9533, 0.9648, 0.9322,
        0.9985, 0.9452, 0.9541, 0.9397, 0.9653, 1.0488, 0.9453, 1.0359, 1.0015, 0.9469, 1.1022, 1.0368,
        0.9572, 0.9159, 1.0379, 1.0832, 1.0534, 1.0536, 1.0482, 1.0556, 0.9516, 1.0317, 0.9518, 1.0323,
        0.9844, 0.9857, 1.0424, 1.1334, 1.0430, 0.9393, 0.9580, 0.9354, 1.0009, 0.9367, 0.9650, 0.9310,
        0.9520, 1.0582, 0.9372, 1.0490, 1.0120, 0.9372, 1.1202, 1.0467, 0.9611, 0.9048, 1.0396, 1.0974
        
]).cuda()
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    wiz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    vis_ndf(decoder, latent, 'vis.exr', resolution=resolution, wiz=wiz)
