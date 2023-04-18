
import numpy as np
import torch
import librosa
import wave
import numpy as np
import scipy
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



x=torch.FloatTensor(3)
print(x)
# x=[]
# x.extend([1,2])
# x.extend([1,2])
# print(x)

# sample_rate = 16000
# window_size = 0.02
# window_stride = 0.01
# n_fft = int(sample_rate * window_size)
# win_length = n_fft
# hop_length = int(sample_rate * window_stride)
# window = "hamming"
# def spectrogram(wav, normalize=True):
#     D = librosa.stft(
#         wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
#     )
#     # 第一维是feature，第二位才是真
#     print(D)
#     print(D.shape)
#     spec, phase = librosa.magphase(D)
#     print(spec)
#     print(spec.shape)
#     print(phase)
#     print(phase.shape)

#     spec = np.log1p(spec)
#     print(spec)
#     print(spec.shape)
#     spec = torch.FloatTensor(spec)
#     print(spec)
#     print(spec.shape)

#     if normalize:
#         spec = (spec - spec.mean()) / spec.std()
#     print(spec)
#     print(spec.shape)
#     return spec


# wav_path="/home/chenyang/chenyang_space/ctc_model_new/data/data_aishell/wav/train/S0002/BAC009S0002W0122.wav"
# with wave.open(wav_path) as wav:
#                 # getnframs返回总帧数  readframes 返回bytes对象 最多给定的参数真 frombuffer以流形式读入数据转化为数组
#                 print(wav.getnframes())
#                 print(wav.getframerate())
#                 wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
#                 print(wav)
#                 print(wav.shape)
#                 wav = wav.astype("float")
#                 print(wav)
#                 print(wav.shape)
#                 wav=(wav-wav.mean())/wav.std()
#                 print(wav)
                
#                 spectrogram(wav)


x[0]=1.0
x[1]=2.0
x[2]=3.0
print(x)
emissions=x
emissions = torch.softmax(emissions,0)
print(emissions)