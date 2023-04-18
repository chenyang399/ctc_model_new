import torch
import librosa
import wave
import numpy as np
import scipy
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
window = "hamming"


def load_audio(wav_path, normalize=True):  # -> numpy array
    with wave.open(wav_path) as wav:
        # getnframs返回总帧数  readframes 返回bytes对象 最多给定的参数真 frombuffer以流形式读入数据转化为数组
        params = wav.getparams()
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        
    # 获取WAV文件的参数
        
        print("params")
        print(params)
        wav = wav.astype("float")
    if normalize:
        return (wav - wav.mean()) / wav.std()
    else:
        return wav


def spectrogram(wav, normalize=True):
    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )

    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    spec = torch.FloatTensor(spec)

    if normalize:
        spec = (spec - spec.mean()) / spec.std()

    return spec


class MASRDataset(Dataset):
    def __init__(self, index_path, labels_path):
        with open(index_path) as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 1) for x in idx]
        self.idx = idx
        with open(labels_path) as f:
            labels = json.load(f)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript = self.idx[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x) for x in transcript]))
        # 返回spect 和翻译，翻译这里用每个字的index表示
        return spect, transcript

    def __len__(self):
        return len(self.idx)


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    # print("in dataloading collate_fn_")
    # print("batch")
    # print(batch)
    # print("batch len")
    # print(len(batch))
    # print("batch len 1")
    # print(len(batch[0]))
    # # print("batch 0")
    # print(batch[0])
    # print(batch[0][0].shape)
    # print(batch[0][1])
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    # 按照sample[0].size(1)来排序
    longest_sample = max(batch, key=func)[0]
    # print("longest sample")
    # print(longest_sample)
    # print("longest sample shape")
    # print(longest_sample.shape)
    # print("bactch[0].size(1)")
    # print(batch[0].size(1))

    # 明白了，这里就是按照每个数据的长度排序，然后把所有的数据都变成最长的长度，方便训练

    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    # print(" minibatch_size = len(batch)")
    # print( minibatch_size = len(batch))
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_lens = torch.IntTensor(minibatch_size)
    # print(inputs)
    # print(inputs.shape)
    # print("input len")
    # print(input_lens)
    # print(input_lens.shape)
    target_lens = torch.IntTensor(minibatch_size)
    # 这里是创建一个tensor不一定都是0，但是之后会把所有的地方都复制，所以没有关系
    # print("target len")
    # print(target_lens)
    # print(target_lens.shape)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0] # wav数据
        target = sample[1] # 翻译数据 这个不是tensor 是list
        seq_length = tensor.size(1)
        
        # print(inputs[x])
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        # print("地界儿  "+str(x))
        # print(inputs[x])
        # print(inputs[x].shape)
        input_lens[x] = seq_length
        target_lens[x] = len(target)
        targets.extend(target) # targets 也就是一个list 这里直接加进去就行,这里是直接放进targets这个list里面，然后我们要用的时候就要结合targetlen来用
    targets = torch.IntTensor(targets)
    
    return inputs, targets, input_lens, target_lens


class MASRDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

