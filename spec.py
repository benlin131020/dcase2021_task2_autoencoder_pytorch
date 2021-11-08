import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from spafe.fbanks import linear_fbanks, mel_fbanks

torchaudio.functional
# Sanity check that indeed we understood the underlying pipeline
wav_name = "../dev_data/gearbox/train/section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"
wav, sr = librosa.load(wav_name, sr=None)
S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=1024, hop_length=512, n_mels=13)

fft_windows = librosa.stft(wav, n_fft=1024, hop_length=512)
magnitude = np.abs(fft_windows)**2
mel = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=13)
# lin = torchaudio.functional.linear_fbanks(n_freqs=magnitude.shape[0], f_min=0, f_max=sr/2.0, n_filter=13, sample_rate=sr)
lin = linear_fbanks.linear_filter_banks(nfilts=13, nfft=1024, fs=sr)
lin_spec = lin.dot(magnitude)
print(lin_spec.shape)
print(S.shape)

# assert (lin.dot(magnitude) == S).all()

# plt.plot()
# plt.show()