import torch
import torchaudio

import os


import torch
import torch.nn as nn
import data
# import home.chenyang.chenyang_space.ctc_model_new.data
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
from torch.nn import CTCLoss
import tensorboardX as tensorboard
import torch.nn.functional as F
import json
import wave
import numpy as np
print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %matplotlib inline

from dataclasses import dataclass

import IPython
import matplotlib
import matplotlib.pyplot as plt





matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
# plt.rcParams.update({'font.size': 16})

torch.random.manual_seed(0)


SPEECH_FILE="/home/chenyang/chenyang_space/ctc_model_new/data/data_aishell/wav/test/S0769/BAC009S0769W0121.wav"
print(SPEECH_FILE)


# /home/chenyang/chenyang_space/ctc_model_new/pretrained_ctc/model_999.pth
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model=torch.load("/home/chenyang/chenyang_space/ctc_model_new/model_998.pth")
model = model.to(device)
with open("/home/chenyang/chenyang_space/ctc_model_new/data/data_aishell/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
labels = vocabulary
print(labels)
print(len(labels))
with torch.inference_mode():
    waveform = data.load_audio(SPEECH_FILE)
    with wave.open(SPEECH_FILE) as wav:
        # getnframs返回总帧数  readframes 返回bytes对象 最多给定的参数真 frombuffer以流形式读入数据转化为数组
        params = wav.getparams()
        wav_raw=np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        # wav_raw = wav_raw.astype("float")
    # params = wav_file.getparams()
    print("waveform")
    print(waveform)
    print(waveform.shape)
    x=data.spectrogram(waveform)
    
    x=torch.unsqueeze(x,0)
    print("waveform")
    print(x)
    print(x.shape)
    x=x.to(device)
    x_len=x.shape[2]
    x_len=torch.tensor(x_len)
    print((x_len))
    emissions,emissions_len = model(x,x_len)
    # emissions = F.log_softmax(emissions, 1)
            # print(out.shape)
    emissions = emissions.transpose(1, 2)
    print("emissions")
    print(emissions)
    print(emissions.shape)
    print("emissions_len")
    print(emissions_len)
    emissions = torch.log_softmax(emissions, dim=-1)
    # print("logsoftmax emissions")
    # print(emissions)
    # print(emissions.shape)

emission = emissions[0].cpu().detach()
print("emission blank prob")
print(emission.T)
print(emission.shape)
print(emission[:,0])

fig, ax = plt.subplots(figsize=(8, 6))

plt.imshow(emission)
plt.colorbar()
plt.title("Frame-wise class probability")
plt.xlabel("Time")
plt.ylabel("Labels")
# plt.show()
ax.set_aspect('auto')
pic_path="./align_pics/1.png"
plt.savefig(pic_path)



# labels_x = dict([(labels[i], i) for i in range(len(labels))])
# labels_str = labels
transcript="该地块即为通州新城核心地标彩虹之门用地"
# transcript = list(filter(None, [labels_x.get(x) for x in transcript]))

# transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
dictionary = {c: i for i, c in enumerate(labels)}

tokens = [dictionary[c] for c in transcript]
print(list(zip(transcript, tokens)))


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    print("cumsum of trellis")
    print(torch.cumsum(emission[:, 0], 0))
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
# 这个函数里面使用的emission里面的t实际上是t+1这里为了使得trellis的第一行和第一列初始化，使得index比较混乱
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


trellis = get_trellis(emission, tokens)


plt.imshow(trellis[1:, 1:].T, origin="lower")
plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
plt.colorbar()
plt.show()


pic_path="./align_pics/2.png"
plt.savefig(pic_path)


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    logging.info("trellis[:, j]")
    logging.info(trellis[:, j])
    t_start = torch.argmax(trellis[:, j]).item()
    logging.info("t_start")
    logging.info(t_start)
    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        logging.info(str(t)+" time step and " +str(j)+" token index in the backtrace algrithum")
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        logging.info("stayed")
        logging.info(stayed)
        logging.info("changed")
        logging.info(changed)
        logging.info("trellis[t - 1, j]")
        logging.info(trellis[t - 1, j])
        logging.info("emission[t - 1, blank_id]")
        logging.info(emission[t - 1, blank_id])
        
        logging.info("trellis[t - 1, j - 1]")
        logging.info(trellis[t - 1, j - 1])
        logging.info("emission[t - 1, tokens[j - 1]]")
        logging.info(emission[t - 1, tokens[j - 1]])
        
        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))
        logging.info("prob")
        logging.info(prob)
        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


path = backtrack(trellis, emission, tokens)
for p in path:
    print(p)



def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")


plot_trellis_with_path(trellis, path)
plt.title("The path found by backtracking")

pic_path="./align_pics/3.png"
plt.savefig(pic_path)

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


segments=[]
prev_token=-1
start = -1
for p in path:
    if p.token_index!=prev_token:
        prev_token=p.token_index
        end=p.time_index
        if start!=-1:
            segments.append(Segment("char", start, end, 1))
        start=p.time_index

# 保存最后一个，start是对的，end是错的
end=path[-1].time_index+1
segments.append(Segment("char", start, end+1, 1))

print("waveform")
print(len(waveform))
print("trellis")
print((trellis.size(0) - 1))

def display_segment(i):
    ratio = len(waveform) / (trellis.size(0) - 1)
    word = segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / 16000:.3f} - {x1 / 16000:.3f} sec")
    segment = wav_raw[x0:x1]
    # 打开新的WAV文件
    with wave.open('align_wavs/new_file_'+str(i)+'.wav', 'wb') as new_wav_file:
        # 设置新文件的参数
        new_wav_file.setparams(params)

        # 写入前5秒的音频数据
        new_wav_file.writeframes(segment)
    # wavfile.write("align_result/"+str(i)+".wav", np.array(segment),16000)
    return IPython.display.Audio(np.array(segment), rate=16000)
# Generate the audio for each segment
print(transcript)
IPython.display.Audio(SPEECH_FILE)


import wave
print(len(tokens))
import scipy.io.wavfile as wavfile
for i in range(len(tokens)):
    print(segments[i])
    audio_obj =display_segment(i)
    # print(audio_obj.data)
    # print(type(audio_obj.data))





