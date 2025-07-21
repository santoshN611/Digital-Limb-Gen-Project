import torch, torch.nn as nn
from monai.networks.nets import DynUNet   # MONAI impl  :contentReference[oaicite:4]{index=4}

def get_dynunet(in_ch=1, out_ch=4):
    return DynUNet(spatial_dims=3,
                   in_channels=in_ch,
                   out_channels=out_ch,
                   kernel_size=[3,3,3,3,3,3],
                   strides=[1,2,2,2,2,1],
                   upsample_kernel_size=[2,2,2,2,2])
