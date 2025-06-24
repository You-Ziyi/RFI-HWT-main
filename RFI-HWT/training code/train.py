import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from models_rddcnn import DnCNN
from models_DNN import DnCNN
import argparse
import utils
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import multiprocessing
multiprocessing.set_start_method('spawn')



device = torch.device("cuda")
# Params
parser = argparse.ArgumentParser(description='DnCNN')
parser.add_argument('--models', default='RDDCNN_', type=str, help='choose a type of models')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch


class SemiSupervisedLoss(nn.Module):
    def __init__(self, height, width, labeled_loss_weight=1.0, unlabeled_loss_weight=0.1):
        super(SemiSupervisedLoss, self).__init__()
        self.labeled_loss_weight = labeled_loss_weight
        self.unlabeled_loss_weight = unlabeled_loss_weight
        self.labeled_criterion = DynamicWeightedMSELoss(height, width)
        self.unlabeled_criterion = nn.MSELoss()  # 一致性损失使用MSELoss

    def forward(self, labeled_outputs, labeled_targets, unlabeled_outputs_1, unlabeled_outputs_2):
        labeled_loss = self.labeled_criterion(labeled_outputs, labeled_targets)
        consistency_loss = self.unlabeled_criterion(unlabeled_outputs_1, unlabeled_outputs_2)
        total_loss = self.labeled_loss_weight * labeled_loss + self.unlabeled_loss_weight * consistency_loss
        return total_loss

class DynamicWeightedMSELoss(nn.Module):
    def __init__(self, height, width):
        super(DynamicWeightedMSELoss, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, 1, height, width))

    def forward(self, inputs, targets):
        weights = torch.sigmoid(self.weights)  # 将权重限制在0到1之间
        loss = weights * (inputs - targets) ** 2
        return loss.mean()



def main():
    print('===> Building models')
    model = DnCNN()
    model.train()
    if cuda:
        print("CUDA is available!")
        model = model.cuda()
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    height, width = 2055, 2055
    semi_supervised_criterion = SemiSupervisedLoss(height, width)
    if cuda:
        semi_supervised_criterion = semi_supervised_criterion.cuda()

    initial_epoch = utils.findLastCheckpoint(save_dir=save_dir)  # load the last models in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.99))
    for epoch in range(initial_epoch, n_epoch):
        xs = dg.datagenerator(data_dir="D:\RFI\clean30")
        ys = dg.datagenerator(data_dir="D:\RFI/noise30")
        zs = dg.datagenerator(data_dir="D:\RFI\data/unlabeled30")

        DDataset = DenoisingDataset(xs)
        DLoader = DataLoader(dataset=DDataset,  drop_last=True, batch_size=batch_size, shuffle=False)
        KDataset = DenoisingDataset(ys)
        KLoader = DataLoader(dataset=KDataset, drop_last=True, batch_size=batch_size, shuffle=False)
        ZDataset = DenoisingDataset(zs)
        ZLoader = DataLoader(dataset=ZDataset, drop_last=True, batch_size=batch_size, shuffle=False)

        labeled_data_iter = iter(zip(DLoader, KLoader))
        unlabeled_data_iter = iter(ZLoader)


        epoch_loss = 0
        start_time = time.time()
        for n_count, (labeled_batch, unlabeled_batch) in enumerate(zip(labeled_data_iter, unlabeled_data_iter)):
            try:
                (batch_x, batch_y) = labeled_batch
            except StopIteration:
                labeled_data_iter = iter(zip(DLoader, KLoader))
                (batch_x, batch_y) = next(labeled_data_iter)
            try:
                batch_z = next(unlabeled_data_iter)
            except StopIteration:
                unlabeled_data_iter = iter(ZLoader)
                batch_z = next(unlabeled_data_iter)

            optimizer.zero_grad()
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_z = batch_z.cuda()

            labeled_outputs = model(batch_y)  # 有标签数据的输出
            unlabeled_outputs_1 = model(batch_z)  # 无标签数据第一次前向传播
            unlabeled_outputs_2 = model(batch_z)  # 无标签数据第二次前向传播（可以添加不同的扰动）

            loss = semi_supervised_criterion(labeled_outputs, batch_x, unlabeled_outputs_1, unlabeled_outputs_2)/batch_size
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        elapsed_time = time.time() - start_time
        scheduler.step()

        utils.log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

if __name__ == "__main__":
    save_dir = os.path.join('models10', 'coeffs30')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    main()
