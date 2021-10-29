from scipy.io import wavfile
import librosa
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]

# class CSCDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         y, sr = librosa.load(self.data[idx], sr=None, mono=False)
#         mel_spectrogram = librosa.feature.melspectrogram(y=y,
#                                                      sr=sr,
#                                                      n_fft=1024,
#                                                      hop_length=512,
#                                                      n_mels=128,
#                                                      power=2.0)
#         # _, song = wavfile.read(self.data[idx]) # Loading your audio
#         return mel_spectrogram

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]