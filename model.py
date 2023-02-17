'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    
    Jiahui Fan, 2022/09
'''

'''
    This script includes definition for the Representation Network and the Layering Network
'''

import torch.nn as nn

class residual(nn.Module):

    def __init__(self, module):
        super(residual, self).__init__()
        self.module = module
        
    def forward(self, inputs):
        return self.module(inputs) + inputs

class layerer_residual_block(nn.Module):

    def __init__(self, input_units=512, output_units=None):
        super(layerer_residual_block, self).__init__()
        if not output_units:
            output_units = input_units
        self.fc1 = nn.Linear(input_units, output_units // 2)
        self.fc2 = nn.Linear(output_units // 2, output_units)
        self.ln1 = nn.LayerNorm(output_units // 2)
        self.ln2 = nn.LayerNorm(output_units)
        self.relu = nn.ReLU()
        if output_units != input_units:
            self.updownsample = nn.Linear(input_units, output_units)

    def forward(self, x):
        output = self.relu(self.ln1(self.fc1(x)))
        if hasattr(self, 'updownsample'):
            x = self.updownsample(x)
        output = self.relu(self.ln2(x + self.fc2(output)))
        return output

'''
    Representation Network
'''
class Decoder_0611_RGB(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_name = type(self).__name__
        self.config = config
        self.init_DR_16()

    def init_DR_16(self):
        self.layer = nn.Sequential(
            self.fclnrelu(32 + 6, 256), # 0 -> 1
            residual(nn.Sequential(
                residual(nn.Sequential(
                    residual(nn.Sequential(
                        layerer_residual_block(256), # 1 -> 2 -> 3
                        self.fclnrelu(256), # 3 -> 4
                        layerer_residual_block(256), # 4 -> 5 -> 6
                    )),
                    self.fclnrelu(256), # 6 -> 7
                    residual(nn.Sequential(
                        layerer_residual_block(256),  # 7 -> 8 -> 9
                        self.fclnrelu(256), # 9 -> 10
                        layerer_residual_block(256), # 10 -> 11 -> 12
                    )),
                )),
                self.fclnrelu(256), # 12 -> 13
                residual(nn.Sequential(
                    residual(nn.Sequential(
                        layerer_residual_block(256), # 13 -> 14 -> 15
                        self.fclnrelu(256), # 15 -> 16
                        layerer_residual_block(256), # 16 -> 17 -> 18
                    )),
                    self.fclnrelu(256), # 18 -> 19
                    residual(nn.Sequential(
                        layerer_residual_block(256), # 19 -> 20 -> 21
                        self.fclnrelu(256), # 21 -> 22
                        layerer_residual_block(256), # 22 -> 23 -> 24
                    )),
                )),
            )),
            self.fclnrelu(256), # 24 -> 25
            nn.Linear(256, 1), # 25 -> 26
        )
            
    def fclnrelu(self, input, output=None):
        if not output:
            output = input
        if self.config.layer_norm:
            return nn.Sequential(
                nn.Linear(input, output, bias=True),
                nn.LayerNorm(output),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input, output, bias=True),
                nn.ReLU(),
            )

    def forward(self, batch_input):
        return self.layer(batch_input)


'''
    Layering Network
'''
class LayererDecoderOnly_alsT_0611_RGB(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_name = type(self).__name__
        self.config = config
        self.init_DR_8()

    def init_DR_8(self):
        self.layer = nn.Sequential(
            self.fclnrelu(32 * 2 + 2, 512),
            residual(nn.Sequential(
                layerer_residual_block(512),
                self.fclnrelu(512),
                residual(nn.Sequential(
                    layerer_residual_block(512),
                    self.fclnrelu(512),
                    layerer_residual_block(512),
                )),
                self.fclnrelu(512),
                layerer_residual_block(512),
            )),
            self.fclnrelu(512, 128),
            nn.Linear(128, 32),
        )

    def fclnrelu(self, input, output=None, leaky=False):
        if not output:
            output = input
        if self.config.layer_layer_norm:
            return nn.Sequential(
                nn.Linear(input, output, bias=True),
                nn.LayerNorm(output),
                nn.ReLU() if not leaky else nn.LeakyReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input, output, bias=True),
                nn.ReLU() if not leaky else nn.LeakyReLU(),
            )

    def forward(self, batch_input):
        '''
            param: batch_input size: [bs, 16 + 16]
        '''
        return self.layer(batch_input)
