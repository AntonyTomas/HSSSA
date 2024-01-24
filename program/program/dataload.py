import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math

#2005-2014
## 五维标准化输入数据集
class MyDataset(Dataset):

    def __init__(self, path, input_seqlen, output_seqlen, channel_num=5, height_num=48, width_num=48,
                 train_precent=0.8, isTrain=True):
        SST = np.load(path)
        #UVSTH
        Temp = SST
        self.temp = Temp
        # print(torch.min(torch.from_numpy(Temp)),torch.max(torch.from_numpy(Temp)))
        self.data_num = len(Temp)
        self.input_seqlen = input_seqlen
        self.output_seqlen = output_seqlen
        self.height_num = height_num
        self.width_num = width_num
        self.channel_num = channel_num
        self.all_seqlen = self.input_seqlen + self.output_seqlen
        self.train_index = int(self.data_num * train_precent)
        self.train = isTrain
        self.data_seq = []
        self.target_seq = []

        for i in range(self.data_num - self.all_seqlen):
            self.data_seq.append(list(range(i, i + self.input_seqlen)))
            self.target_seq.append(list(range(i + self.input_seqlen, i + self.all_seqlen)))

        if self.train:
            self.data_seq = self.data_seq[:self.train_index]
            self.target_seq = self.target_seq[:self.train_index]

        else:
            self.data_seq = self.data_seq[self.train_index:]
            self.target_seq = self.target_seq[self.train_index:]

        self.data_seq = np.array(self.data_seq)  # .reshape((len(self.data_seq), -1, channel_num, height_num, width_num))
        self.target_seq = np.array(self.target_seq)  # .reshape(

    def __getitem__(self, index):
        self.input_sample = self.temp[self.data_seq[index]]

        self.input_sample = (self.input_sample[:,0:1,:,:])/31.23

        self.output_sample = self.temp[self.target_seq[index]]
        self.output_sample = self.output_sample[:, 0:1, :, :]
        return self.input_sample, self.output_sample

    def __len__(self):
        return len(self.data_seq)


if __name__ == '__main__':
    path = r'D:\workspace\Ydq\SAMHSA\data_use.npy'
    data = np.load(path)
    data_max = np.max(data)
    data_min = np.min(data)
    data = data/31.23
    # data_max1 = np.max(data)
    # data_min1 = np.min(data)

    # UVSTH
    # data_mat = np.load(path)[-3650:,3:4,:,:]
    # minval = np.min(data_mat[np.nonzero(data_mat)])
    # maxval = np.max(data_mat[np.nonzero(data_mat)])


    data = MyDataset(path, input_seqlen=7, output_seqlen=7)
    inp, tar = data[1001]
    print(inp.shape, tar.shape)



0