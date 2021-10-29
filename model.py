import torch
import torch.nn as nn
from nnAudio import Spectrogram

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.bottleneck_layer = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.after_bottleneck_layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.output = nn.Linear(128, self.input_dim)

        self.encoder_layers = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()) for _ in range(3)])
        

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.encoder_layers:
            x = l(x)
        x = self.bottleneck_layer(x)
        x = self.after_bottleneck_layer(x)
        for l in self.decoder_layers:
            x = l(x)
        x = self.output(x)
        return x

class AutoEncoderSpec(nn.Module):
    def __init__(self, mels, fft, hop, frames):
        super(AutoEncoderSpec, self).__init__()
        self.input_dim = mels * frames
        self.spec_layer = Spectrogram.MelSpectrogram(n_fft=fft, n_mels=mels,
                                           hop_length=hop, window='hann', sr=48000)
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.bottleneck_layer = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.after_bottleneck_layer = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.output = nn.Linear(128, self.input_dim)

        self.encoder_layers = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()) for _ in range(3)])
        

    def forward(self, x):
        x = self.spec_layer(x)
        print(x.shape)
        x = self.input_layer(x)
        for l in self.encoder_layers:
            x = l(x)
        x = self.bottleneck_layer(x)
        x = self.after_bottleneck_layer(x)
        for l in self.decoder_layers:
            x = l(x)
        x = self.output(x)
        return x