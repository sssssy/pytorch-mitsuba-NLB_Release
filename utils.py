'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.

    Jiahui Fan, 2022/09
'''

'''
	This script includes network configurations and helper functions
'''

import random, os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

GLOBAL_RANDOM_SEED = datetime.now().microsecond
torch.manual_seed(GLOBAL_RANDOM_SEED)
torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)
torch.backends.cudnn.deterministic = True
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
torch.set_printoptions(sci_mode=False, linewidth=110)
np.set_printoptions(suppress=True)

class Config():

	def to_lines(self):
		lines = []
		for k,v in vars(self).items():
			lines.append('{:<30} {}'.format(k, v))
		return lines
		
	def print(self):
		for line in self.to_lines():
			print(line)
		print()

'''
	Representation Network
'''
class DecoderOnlyConfig(Config):

    def __init__(self):
        
        super().__init__()
        self.latent_size = 32 * 3
        self.layer_norm = True
        self.is_cuda = torch.cuda.is_available()
        print(f'>>> cuda: {self.is_cuda}')
        self.compress_decoder = 'Decoder_0611_RGB'
        self.decoder_path = './saved_model/RepreNetwork.pth'

'''
	Layering Network
'''
class LayerDecoderOnly_alsTN_Config(Config):

    def __init__(self):

        self.input_latent_size = 32
        self.latent_size = 32
        self.layer_layer_norm = True
        self.is_cuda = torch.cuda.is_available()
        print(f'>>> cuda: {self.is_cuda}')
        self.layer_model = 'LayererDecoderOnly_alsT_0611_RGB'
        self.layerer_path = './saved_model/LayeringNetwork.pth'

'''
	Helper function to convert (view_x, view_y, light_x, light_y) to (view_x, view_y, light_x, light_y, view_z, light_z)

 Args:
	wiwo: torch.tensor (batch_size, N, 4)
'''
def to6d(wiwo):
    wiz = torch.sqrt(torch.clamp(1 - wiwo[:, :, 0:1]**2 - wiwo[:, :, 1:2]**2, min=0))
    woz = torch.sqrt(torch.clamp(1 - wiwo[:, :, 2:3]**2 - wiwo[:, :, 3:4]**2, min=0))
    return torch.cat([wiwo, wiz, woz], axis=-1)

'''
	Helper function to format time from seconds to %H:%M:%S
'''
def time_change(time_input):
	time_list = []
	if time_input/3600 > 1:
		time_h = int(time_input/3600)
		time_m = int((time_input-time_h*3600) / 60)
		time_s = int(time_input - time_h * 3600 - time_m * 60)
		time_list.append(str(time_h))
		time_list.append('h ')
		time_list.append(str(time_m))
		time_list.append('m ')

	elif time_input/60 > 1:
		time_m = int(time_input/60)
		time_s = int(time_input - time_m * 60)
		time_list.append(str(time_m))
		time_list.append('m ')
	else:
		time_s = int(time_input)

	time_list.append(str(time_s))
	time_list.append('s')
	time_str = ''.join(time_list)
	return time_str