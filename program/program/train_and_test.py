import sys
from torch.utils.data import DataLoader

sys.path.insert(0, '../')
import torch
import torch.nn as nn
from dataload import MyDataset
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os.path as osp
import os
import shutil
from criterion import GDL

from encoder_decoder import EncoderFun, DecoderFun, EF
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = r'J:\ydq\20230925_final\SAMHA\data_use.npy'
nan_mask = np.load(r'J:\ydq\20230925_final\SAMHA\nan_mask.npy')
nan_mask = torch.from_numpy(nan_mask)
nan_mask = nan_mask.to(device)

# check & build the files
save_dir = r'J:\ydq\20230925_final\different_input\15_15\SST\save_iter_use'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
model_save_dir = osp.join(save_dir, 'models')
log_dir = osp.join(save_dir, 'logs')
all_scalars_file_name = osp.join(save_dir, "all_scalars.json")
# pkl_save_dir = osp.join(save_dir, 'pkl')
if osp.exists(all_scalars_file_name):
    os.remove(all_scalars_file_name)
if osp.exists(log_dir):
    shutil.rmtree(log_dir)
if osp.exists(model_save_dir):
    shutil.rmtree(model_save_dir)
os.mkdir(model_save_dir)

writer = SummaryWriter(log_dir)
train_mse = 0
train_gdl = 0
train_loss = 0
batch_size = 4

in_seqlen =15
out_seqlen = 15
max_iterations = 500
train_data = MyDataset(path, input_seqlen=in_seqlen, output_seqlen=out_seqlen)
test_data = MyDataset(path, input_seqlen=in_seqlen, output_seqlen=out_seqlen, isTrain=False)

test_data_loader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              shuffle=False)

# 构建网络
encFun = EncoderFun(in_channels=1, kernel_size=3, stride=1)
encFun.to(device)
encBlock1 = EncoderBlock(
    encH=48,
    encW=48,
    channels=64,
    num_heads=8,
    K=8,
    dropout=0.,
    drop_path=0.,
    Spatial_FFN_hidden_ratio=4,
    dim_feedforward=256,
)
encBlock1.to(device)

encBlock2 = EncoderBlock(
    encH=48,
    encW=48,
    channels=64,
    num_heads=8,
    K=8,
    dropout=0.,
    drop_path=0.,
    Spatial_FFN_hidden_ratio=4,
    dim_feedforward=256,
)
encBlock2.to(device)

decBlock2 = DecoderBlock(
    encH=48,
    encW=48,
    channels=64,
    num_heads=8,
    K=8,
    dropout=0.,
    drop_path=0.,
    Spatial_FFN_hidden_ratio=4,
    dim_feedforward=256
)
decBlock2.to(device)

decBlock1 = DecoderBlock(
    encH=48,
    encW=48,
    channels=64,
    num_heads=8,
    K=8,
    dropout=0.,
    drop_path=0.,
    Spatial_FFN_hidden_ratio=4,
    dim_feedforward=256
)
decBlock1.to(device)
decFun = DecoderFun(out_channels=1, kernel_size=3, stride=1)
decFun.to(device)



encoder_decoder = EF(encoderfun=encFun, encblc1=encBlock1,encblc2=encBlock2, decblc2=decBlock2,decblc1=decBlock1,decoderfun=decFun, )
encoder_decoder.to(device)

mse_loss = nn.MSELoss().to(device)
GDL_loss = GDL().to(device)

gamma = 0.1
LR = 1e-3
weight_decay = 1e-4
optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=LR, weight_decay=weight_decay)
exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=10, eps=1e-6)

for itera in tqdm(range(1, max_iterations + 1)):

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True)

    encoder_decoder.train()
    for i, (trained_data, trained_label) in enumerate(train_data_loader):
        trained_data = trained_data.to(device)
        trained_label = trained_label.to(device)
        output = encoder_decoder(trained_data) *31.23* nan_mask
        MSE_iter = mse_loss(output, trained_label)
        GDL_iter = GDL_loss(output, trained_label)
        loss = MSE_iter+GDL_iter
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_mse += MSE_iter.item()
        train_gdl += GDL_iter.item()
        train_loss += loss.item()

    if itera % 1 == 0:

        train_mse_print = train_mse / 1 / len(train_data_loader)
        train_gdl_print = train_gdl / 1 / len(train_data_loader)
        train_loss_print = train_loss / 1 / len(train_data_loader)
        print('Train_mse', train_mse_print)
        print('Train_gdl', train_gdl_print)
        print('Train_loss', train_loss_print)
        train_mse = 0.0
        train_gdl = 0.0
        train_loss = 0.0
        valid_mse = 0.0
        valid_gdl = 0.0
        valid_loss = 0.0
        encoder_decoder.eval()
        ############
        with torch.no_grad():
            for i, (test_data, test_label) in enumerate(test_data_loader):
                valid_data = test_data.to(device)
                valid_label = test_label.to(device)
                valid_output = encoder_decoder(valid_data)*31.23* nan_mask

                v_MSE_iter = mse_loss(valid_output, valid_label)
                v_GDL_iter = GDL_loss(valid_output, valid_label)
                v_loss = v_MSE_iter + v_GDL_iter

                valid_mse += v_MSE_iter.item()
                valid_gdl += v_GDL_iter.item()
                valid_loss += v_loss.item()

            valid_mse_print = valid_mse / len(test_data_loader)
            valid_gdl_print = valid_gdl / len(test_data_loader)
            valid_loss_print = valid_loss / len(test_data_loader)
            print('valid_mse', valid_mse_print)
            print('valid_gdl', valid_gdl_print)
            print('valid_loss', valid_loss_print)

        torch.optim.lr_scheduler
        exp_lr_scheduler.step(valid_loss_print)
        #####################################
        writer.add_scalars("mse", {
            "train": train_mse_print,
            "valid": valid_mse_print
        }, itera)
        writer.add_scalars("gdl", {
            "train": train_gdl_print,
            "valid": valid_gdl_print
        }, itera)
        writer.add_scalars("loss", {
            "train": train_loss_print,
            "valid": valid_loss_print
        }, itera)

        writer.add_scalars("lr", {
            "lr": optimizer.state_dict()['param_groups'][0]['lr'],
        }, itera)

        writer.export_scalars_to_json(all_scalars_file_name)

    if itera % 1 == 0:
        torch.save(encoder_decoder.state_dict(),
                   osp.join(model_save_dir, 'encoder_forecaster_{}.pth'.format(itera)))

    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    # print(lr_scheduler.get_lr()[0])
writer.close()

# def plot_result(writer, itera, train_result, valid_result):
#     train_mse = train_result
#     train_mse = np.nan_to_num(train_mse)
#
#     valid_mse = valid_result
#     valid_mse = np.nan_to_num(valid_mse)
#
#     writer.add_scalars("mse", {
#         "train": train_mse.mean(),
#         "valid": valid_mse.mean(),
#         "train_last_frame": train_mse[-1],
#         "valid_last_frame": valid_mse[-1],
#     }, itera)
