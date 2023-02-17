'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    	
    Jiahui Fan, 2022/09
'''

'''
    This module is a wrapper of OpenEXR, including some calculation and manipulation functions.
    Added and edited by Jiahui Fan.
'''

import OpenEXR, Imath
import numpy as np

def info(filename):
    print('========================================')
    print(filename)
    print()
    img = OpenEXR.InputFile(filename)
    for k, v in img.header().items():
        print(f'{k:<20}', v, sep='\t')
    print('========================================')


def read(filename, channels=None, collapse=True, verbose=False):

    '''
    read an OpenEXR image.\n
    the image is single-channel (gray), 3-channel (RGB) or 4-channel (RGBA) by default.\n

    Args:
        filename (str):     the filename of the OpenEXR image.\n
        channels (list):    an ORDERED list of channels to read. Default: 1 channel: [gray], 3 channels: [R, G, B] or 4 channels: [R, G, B, A]\n
        collapse (bool):    whether to collapse the RGBA into RGB image. Default: True.\n
        verbose  (bool):    print info() if True. Default: False.\n

    Returns:
        np.ndarray: the image. the first dim of img.shape is always vertical (H) in display,\n
                    corresponding to x-axis in np.ndarray and y-axis in Header.dataWindow\n
    '''
    if verbose:
        info(filename)

    img = OpenEXR.InputFile(filename)
    header = img.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1) ##! the xy is transposed in dataWindow.
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    def chan(c):
        s = img.channel(c, pt)
        arr = np.frombuffer(s, dtype=np.float32)
        arr.shape = size
        return arr
    
    num_channels = len(header['channels'].keys()) if channels is None else len(channels)

    if channels is None and num_channels not in [1, 3, 4]:
        raise ValueError('The image is not single-channel, 3-channel or 4-channel and $channels is not specified.')

    if channels is None and num_channels in [1, 3, 4]:
        # single-channel file
        if num_channels == 1:
            return chan(header['channels'].keys()[0])

        # standard RGB file, collapse RGBA into RGB for common use
        if num_channels == 3 or (num_channels == 4 and collapse):
            return np.dstack([chan('R'), chan('G'), chan('B')])
        
        if num_channels == 4 and not collapse:
            return np.dstack([chan('R'), chan('G'), chan('B'), chan('A')])

    if channels is not None and set(channels) != set(header['channels'].keys()):
        raise ValueError(f'channels mismatch: {channels} != {header["channels"].keys()}')
    
    ## channels provided (maybe in [1, 3, 4] but not the default configurations)
    res = []
    for key in channels:
        res.append(chan(key))
    return np.dstack(res)

    
def writeRGB(img, filename):
    '''
    write np.ndarray to OpenEXR image, converting into np.float32 firstly.\n
    recive a image of size (h, w), (h, w, 1) or (h, w, 3).\n
    always write into a 3-channel RGB file.\n
    the first dim of img.shape is always vertical (H) in display,\n
    corresponding to x-axis in np.ndarray and y-axis in Header.dataWindow\n

    Args:
        img (np.ndarray):   img data to write.\n
        filename (str):     the filename of the OpenEXR image.\n
    '''

    if img.dtype is not np.float32:
        img = img.astype(np.float32)

    h, w = img.shape[:2]

    header = OpenEXR.Header(w, h) ##! the xy is transposed in dataWindow
    header['Compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    out = OpenEXR.OutputFile(filename, header)

    num_channels = 1 if len(img.shape) == 2 else img.shape[2]

    # single-channel file
    if num_channels == 1:
        img = img.reshape(h, w)
        out.writePixels({'R': img.tostring(), 'G': img.tostring(), 'B': img.tostring()})
        return

    # standard RGB file or RGBA file
    if num_channels == 3:
        out.writePixels({'R': img[:,:,0].tostring(), 'G': img[:,:,1].tostring(), 'B': img[:,:,2].tostring()})
        return

    raise ValueError('The image is not single-channel or 3-channel.')

    
def writeRGB16(img, filename):
    '''
    write np.ndarray to OpenEXR image, converting into np.float16 firstly.\n
    recive a image of size (h, w), (h, w, 1) or (h, w, 3).\n
    always write into a 3-channel RGB file.\n
    the first dim of img.shape is always vertical (H) in display,\n
    corresponding to x-axis in np.ndarray and y-axis in Header.dataWindow\n

    Args:
        img (np.ndarray):   img data to write.\n
        filename (str):     the filename of the OpenEXR image.\n
    '''

    if img.dtype is not np.float16:
        img = img.astype(np.float16)

    h, w = img.shape[:2]

    header = OpenEXR.Header(w, h) ##! the xy is transposed in dataWindow
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(filename, header)

    num_channels = 1 if len(img.shape) == 2 else img.shape[2]

    # single-channel file
    if num_channels == 1:
        img = img.reshape(h, w)
        out.writePixels({'R': img.tostring(), 'G': img.tostring(), 'B': img.tostring()})
        return

    # standard RGB file or RGBA file
    if num_channels == 3:
        out.writePixels({'R': img[:,:,0].tostring(), 'G': img[:,:,1].tostring(), 'B': img[:,:,2].tostring()})
        return

    raise ValueError('The image is not single-channel or 3-channel.')


def write32(img, filename):
    writeRGB(img, filename)

def write(img, filename):
    writeRGB(img, filename)

def writeMono32(img, filename):
    writeRGB(img, filename)