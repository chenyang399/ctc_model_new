


import os


import torch
import torch.nn as nn
import data
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
from torch.nn import CTCLoss
import tensorboardX as tensorboard
import torch.nn.functional as F
import json







def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            # print("in decoding")
            # print("xshape")
            # print(x.shape)
            # print("x lens")
            # print(x_lens)
            # print("x")
            # print(x)
            # print("yshape")
            # print(y.shape)
            # print("y lens")
            # print(y_lens)
            # print("y")
            # print(y)
            x = x.to("cuda")
            outs, out_lens = model(x, x_lens)
            # print("out shape")
            # print(outs.shape)
            # print("outlen")
            # print(out_lens)

            outs = F.softmax(outs, 1) # 按照维度1进行归一化
            # print("outs shape after softmax")
            # print(outs.shape)
            outs = outs.transpose(1, 2)
            # print("outs shape after transpose")
            # print(outs.shape)

            
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset : offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) / float(len(ref))
                print("in cer cal")
                print("pred")
                print(pred)
                print("truth")
                print(truth)
                print("trans")
                print(trans)
                print("ref")
                print(ref)
                print("decoder.cer(trans, ref)")
                print(decoder.cer(trans, ref))
                print("float(len(ref))")
                print(float(len(ref)))
               

             

                
        cer /= len(dataloader.dataset)
    model.train()
    return cer


batch_size=64
train_index_path="data/data_aishell/train-sort.manifest"
dev_index_path="data/data_aishell/dev.manifest"
labels_path="data/data_aishell/labels.json"
model=torch.load( "model_lr=0.6_343.pth")
train_dataset = data.MASRDataset(train_index_path, labels_path)
dev_dataset = data.MASRDataset(dev_index_path, labels_path)
train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8
    )
train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
    )
dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=8
    )

eval(model,dev_dataloader)

