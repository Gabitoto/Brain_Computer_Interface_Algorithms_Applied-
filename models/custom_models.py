"""
Definición de arquitectura EEGNet para clasificación de señales EEG
Basada en la arquitectura propuesta por Lawhern et al. (2018)
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNetv4


class EEGNet(nn.Module):
    """
    EEGNet: Red neuronal convolucional para clasificación de señales EEG
    Basada en la arquitectura propuesta por Lawhern et al. (2018)
    """
    def __init__(self, n_classes=4, n_chans=22, n_times=1000, sfreq=250,
                 F1=8, F2=16, D=2, kernel_length=64, pool_time_length=4, pool_time_stride=4):
        super(EEGNet, self).__init__()
        
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.n_times = n_times
        
        # Usar EEGNetv4 de Braindecode
        self.model = EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=None,
            n_filters_time=F1,
            filter_time_length=kernel_length,
            n_filters_spat=F1 * D,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            n_filters_2=F2,
            filter_length_2=kernel_length // 2,
            n_filters_3=n_classes,
            filter_length_3=kernel_length // 4,
            final_conv_length='auto',
            pool_time_length_2=pool_time_length,
            pool_time_stride_2=pool_time_stride,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
        )
    
    def forward(self, x):
        return self.model(x)
