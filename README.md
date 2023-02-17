# Overview

These are experimental scripts for SIGRAPPH 2022 paper ``Neural Layered BRDFs``. In this paper, we proposed to use neural networks to compress and evaluate BRDFs, as well as to layer them in a projected latent space. 

> Project Homepage: https://wangningbei.github.io/2022/NLBRDF.html  
> My Homepage: https://whois-jiahui.fun

Please feel free to try out these codes to COMPRESS or LAYER your own BRDFs of any kinds (as long as they're in expected data formats).

**NOTE**: These scripts are suboptimal and experimental. There may be redundant lines and functionalities. These codes are provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. One can arbitrarily change or redistribute these scripts with above statements.

# Requirements

pytorch (tested on ver-1.2.0)  
openexr (tested on ver-1.3.2)

# Contents

```
DIR
--
  \-    data/                           - Directory for BRDF data to compress
   |-       0-0.npy                         - An example metal BRDF
   |-       0-0-bsdf_0.exr                  - Data visualization for 0-0.npy
   |-       1-0.npy                         - An example rough dielectric BRDF
   `-       1-0-bsdf_0.exr                  - Data visualization for 1-0.npy
  \-    saved_model/                    - Directory for pretrained model to use
   |-       RepreNetwork.pth                - The Representation Network
   `-       LayeringNetwork.pth             - The Layering Network
  |-    README.md                       - Here I am!
  |-    exr.py                          - A tool script to save and load OpenEXR files
  |-    model.py                        - Model definitions
  |-    utils.py                        - Model configs and some helper functions
  |-    compress.py                     * Script to compress a given BRDF data (.npy)
  |-    layering.py                     * Script to layer 2 given BRDF latent codes
  `-    visualize.py                    * Script to visualize any compressed BRDF latent code
```

# Quick Start

*All scripts have their individual help documents and detailed usage. You can check them and make your own modifications.*

Here are some basic examples:

1. To compress a given BRDF data in ``.npy`` file format (the detailed format structure will be described [here](#format)), run:

```
$ python compress.py ./data/0-0.npy
torch.Size([1, 390625, 7])
>>> cuda: True
>>> step 1000, avg_loss: 0.0072 +- (0.0051), latent: 0.9955, lr: 0.0000153
tensor([[[1.0357, 1.0408, 1.0111, 1.0082, 0.9337, 1.0371, 0.9698, 1.0472, 1.0252, 0.9581, 0.9941, 0.9998,
          1.0237, 0.9839, 0.9645, 0.9899, 1.0191, 0.9807, 0.9569, 1.0062, 0.9887, 1.0228, 0.9749, 0.9871,
          0.9320, 1.0019, 0.9645, 1.0687, 1.0633, 1.0077, 1.0002, 1.0204, 1.0279, 0.9733, 0.9622, 0.9880,
          0.9869, 0.9796, 0.9412, 1.0024, 0.9851, 1.0293, 0.9760, 1.0081, 0.9759, 0.9985, 0.9706, 0.9612,
          0.9785, 0.9028, 1.0155, 1.0274, 1.0375, 1.0488, 1.0276, 1.0250, 0.9497, 0.9905, 0.9466, 1.0764,
          1.1145, 1.0373, 1.0133, 1.0048, 1.0359, 0.9605, 0.9625, 0.9859, 0.9715, 0.9764, 0.9293, 0.9847,
          0.9677, 1.0352, 0.9703, 1.0174, 0.9892, 0.9832, 1.0055, 0.9529, 0.9692, 0.9253, 1.0243, 1.0123]]],
```

2. To layer 2 compressed BRDF latent code, firstly, you need to fill out all the tensor values *(both the top layer and bottom layer)* and the media parameters *(following the definitions in our paper)* in ``layering.py``. Then, directly run:

```
$ python layering.py
>>> cuda: True
tensor([1.0504, 1.0140, 0.9858, 0.9929, 0.9551, 1.0667, 1.0301, 0.9681, 0.9994, 0.9346, 0.9755, 1.0965,
        0.9729, 0.9676, 1.0273, 0.9618, 1.0504, 0.9365, 1.0141, 0.9615, 1.0063, 0.9993, 0.9564, 1.0172,
        0.9706, 0.9159, 1.0595, 1.0985, 1.0210, 0.9151, 0.9846, 1.1005, 1.0531, 1.0070, 0.9834, 0.9946,
        0.9427, 1.0695, 1.0329, 0.9671, 1.0030, 0.9307, 0.9717, 1.0957, 0.9655, 0.9536, 1.0258, 0.9603,
        1.0166, 0.9141, 0.9857, 1.0903, 1.0593, 1.0096, 0.9855, 0.9977, 0.9450, 1.0692, 1.0376, 0.9607,
        0.9988, 0.9997, 0.9404, 1.0150, 0.9876, 0.9113, 1.0639, 1.0914, 1.0155, 0.9163, 0.9827, 1.0837],
       device='cuda:0')
```

3. To visualize any compressed (or layered) BRDF latent code, fill it into ``visualize.py`` and run:

```
$ python visualize.py
>>> cuda: True
```


<h1 id="format">Data Formats</h1>

All BRDF queries (don't need to be ordered) are stored in ``.npy`` files. It should contain an ``np.float32`` array of size `(1, num_queries, 7)`. 

For each query, the 7 entries are `(view_x, view_y, light_x, light_y, R, G, B)`. View (light) coordinates are normalized.

Example: 0-0.npy
```
array([[ 0.02173355,  0.00358327,  0.06159558, ...,  0.33209467,  0.23438323,  0.15506488],
       [ 0.05040421,  0.00568237,  0.00537076, ...,  0.44599196,  0.30289957,  0.18345068],
       [ 0.03463246,  0.00795274,  0.03160701, ...,  0.40654963,  0.26484686,  0.1737824 ],
       ...,
       [ 0.9930808 , -0.11743117,  0.76401556, ...,  0.0003436 ,  0.00026537,  0.00018851],
       [ 0.9935267 , -0.11252784,  0.9467221 , ...,  0.00004106,  0.00003232,  0.00002456],
       [ 0.983361  , -0.17322123,  0.9923124 , ...,  0.00034331,  0.00027223,  0.0002034 ]], dtype=float32)
```

# BibTex

Please cite our paper for any usage of our code in your work by
```
@inproceedings{Fan:2022:NLBRDF,
  title={Neural Layered BRDFs},
  author={Jiahui Fan and Beibei Wang and Milo\v{s} Ha\v{s}an and Jian Yang and Ling-Qi Yan},
  booktitle={Proceedings of SIGGRAPH 2022},
  year={2022}
}
```