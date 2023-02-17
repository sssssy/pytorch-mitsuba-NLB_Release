'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    	
    Jiahui Fan, 2022/09
'''

'''
    This script layer 2 given latent codes into a new latent code
    There're example latent codes, please replace them with your own BRDF latent codes.
'''

import sys

import torch
import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
torch.set_printoptions(sci_mode = False, linewidth=120)

import model
from utils import *

if __name__ == '__main__':
    
    if len(sys.argv) != 1:
        print(
    '''
        Usage: $ python layering.py
        
        Args:
            None
    '''
        )
        exit()

    config = LayerDecoderOnly_alsTN_Config()

    layerer = getattr(model, config.layer_model)(config)
    layerer.load_state_dict(torch.load(config.layerer_path)())
    layerer = layerer.cuda().eval()

    latent = torch.Tensor([

## ''' top layer latent '''
1.0612, 1.0654, 1.0614, 1.0730, 0.9225, 1.0260, 0.9427, 1.0506, 0.9631, 1.0320, 1.0695, 1.0979,
          1.0565, 0.9360, 0.9368, 0.9340, 0.9689, 0.9265, 0.9611, 0.9153, 0.9335, 1.0632, 0.9161, 1.0621,
          1.0503, 0.9159, 1.0876, 1.0519, 0.9537, 0.8968, 1.0690, 1.0926, 1.0612, 1.0654, 1.0614, 1.0730,
          0.9225, 1.0260, 0.9427, 1.0506, 0.9631, 1.0320, 1.0695, 1.0979, 1.0565, 0.9360, 0.9368, 0.9340,
          0.9689, 0.9265, 0.9611, 0.9153, 0.9335, 1.0632, 0.9161, 1.0621, 1.0503, 0.9159, 1.0876, 1.0519,
          0.9537, 0.8968, 1.0690, 1.0926, 1.0612, 1.0654, 1.0614, 1.0730, 0.9225, 1.0260, 0.9427, 1.0506,
          0.9631, 1.0320, 1.0695, 1.0979, 1.0565, 0.9360, 0.9368, 0.9340, 0.9689, 0.9265, 0.9611, 0.9153,
          0.9335, 1.0632, 0.9161, 1.0621, 1.0503, 0.9159, 1.0876, 1.0519, 0.9537, 0.8968, 1.0690, 1.0926,

## ''' bottom layer latent '''
1.0357, 1.0408, 1.0111, 1.0082, 0.9337, 1.0371, 0.9698, 1.0472, 1.0252, 0.9581, 0.9941, 0.9998,
          1.0237, 0.9839, 0.9645, 0.9899, 1.0191, 0.9807, 0.9569, 1.0062, 0.9887, 1.0228, 0.9749, 0.9871,
          0.9560, 1.0002, 0.9644, 0.9690, 0.9858, 0.8786, 1.0145, 1.0371, 1.0406, 1.0501, 1.0168, 1.0147,
          0.9320, 1.0019, 0.9645, 1.0687, 1.0633, 1.0077, 1.0002, 1.0204, 1.0279, 0.9733, 0.9622, 0.9880,
          0.9869, 0.9796, 0.9412, 1.0024, 0.9851, 1.0293, 0.9760, 1.0081, 0.9759, 0.9985, 0.9706, 0.9612,
          0.9785, 0.9028, 1.0155, 1.0274, 1.0375, 1.0488, 1.0276, 1.0250, 0.9497, 0.9905, 0.9466, 1.0764,
          1.1145, 1.0373, 1.0133, 1.0048, 1.0359, 0.9605, 0.9625, 0.9859, 0.9715, 0.9764, 0.9293, 0.9847,
          0.9677, 1.0352, 0.9703, 1.0174, 0.9892, 0.9832, 1.0055, 0.9529, 0.9692, 0.9253, 1.0243, 1.0123,

## ''' albedo (R, G, B) | sigmaT ''''
1.0, 1.0, 1.0, 0.0

    ]).cuda().reshape(-1)
    
    layer_input1 = torch.cat([
        latent[0:32], latent[96: 96+32], latent[-4:-3] +1, latent[-1:] +1 #! norm
    ])
    layer_input2 = torch.cat([
        latent[32:64], latent[96+32: 96+64], latent[-3:-2] +1, latent[-1:] +1
    ])
    layer_input3 = torch.cat([
        latent[64:96], latent[96+64: 96+96], latent[-2:-1] +1, latent[-1:] +1
    ])

    layer_output1 = layerer(layer_input1-1).detach().reshape(1, 1, 32) +1
    layer_output2 = layerer(layer_input2-1).detach().reshape(1, 1, 32) +1
    layer_output3 = layerer(layer_input3-1).detach().reshape(1, 1, 32) +1
    
    output_latent = torch.cat([layer_output1, layer_output2, layer_output3], axis=-1).reshape(-1)
    print(output_latent)
