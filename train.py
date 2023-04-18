
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


def train(
    model,
    epochs=1000,
    batch_size=64,
    train_index_path="data/data_aishell/train-sort.manifest",
    dev_index_path="data/data_aishell/dev.manifest",
    labels_path="data/data_aishell/labels.json",
    learning_rate=0.6,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    
    
    save_path = './log/'

    print("start dataset")
    train_dataset = data.MASRDataset(train_index_path, labels_path)
  
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    print("end dataset")
    print("start dataloader")
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=8
    )
    print("end dataloader")
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    ctcloss = CTCLoss(blank=0, reduction='sum',zero_infinity=True)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    writer = tensorboard.SummaryWriter(save_path)
    gstep = 0
    # last_epoch=14
    for epoch in range(epochs):
        # torch.save(model, "pretrained/model_{}.pth".format(epoch))
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # lr_sched.step()
        lr = get_lr(optimizer)
        writer.add_scalar("lr/epoch", lr, epoch)
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.to("cuda")
            # print("x")
            # print(x)
            # print(x.shape)
            # print("x_len")
            # print(x_lens)
            # print(x_lens.shape)
            # print("y")
            # print(y)
            # print(y.shape)
            # print("y_len")
            # print(y_lens)
            # print(y_lens.shape)


            out, out_lens = model(x, x_lens)
            out = F.log_softmax(out, 1)
            # print(out.shape)
            out = out.transpose(0, 1).transpose(0, 2)
            loss = ctcloss(out, y, out_lens, y_lens)/out.size(1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("loss/step", loss.item(), gstep)
            gstep += 1
            print(
                "[{}/{}][{}/{}]\tLoss = {}".format(
                    epoch + 1, epochs, i, int(batchs), loss.item()
                )
            )
            # cer = eval(model, dev_dataloader)
        epoch_loss = epoch_loss / batchs
        cer = eval(model, dev_dataloader)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        writer.add_scalar("cer/epoch", cer, epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        torch.save(model, "pretrained_ctc/model_{}.pth".format(epoch))
        # torch.save(model, "pretrained/model_lr=0.6.pth")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


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
                # print("in cer cal")
                # print("pred")
                # print(pred)
                # print("truth")
                # print(truth)
                # print("trans")
                # print(trans)
                # print("ref")
                # print(ref)
                # print("decoder.cer(trans, ref)")
                # print(decoder.cer(trans, ref))
                # print("float(len(ref))")
                # print(float(len(ref)))
               

             

                
        cer /= len(dataloader.dataset)
    model.train()
    return cer


if __name__ == "__main__":
    with open("data/data_aishell/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
        # print(vocabulary)
    print("start training 1")
    # torch.backends.cudnn.enabled = False
    model = GatedConv(vocabulary)

    print("start training 2")
   
    model.to("cuda")
    print("start training 3")
    
    x=torch.randn(100)
    y=x.cuda()
    print(type(y))
    print((y))
    train(model)
