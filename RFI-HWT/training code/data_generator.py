import glob
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from torch.utils.data import Dataset

#给数据增加一个维度，满足pytorch格式
class DenoisingDataset(Dataset):
    def __init__(self, xs):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
    def __getitem__(self, index):
        batch_x = self.xs[index]
        # 添加一个维度表示通道数
        batch_x = batch_x.unsqueeze(0)
        return batch_x
    def __len__(self):
        return self.xs.size(0)

#数据处理：转成张量格式、进行数据归一化
def datagenerator(data_dir='None', channels=1):
    file_list = glob.glob(data_dir+'/*.npy')
    data = []
    for i in range(len(file_list)):
        img = np.load(file_list[i])
        scaler = RobustScaler()
        data_singal = scaler.fit_transform(img)
        data.append(data_singal)
    data_array = np.array(data)
    data_tensor = torch.from_numpy(data_array).float()
    print('^_^-training data prepared-^_^')
    return data_tensor

if __name__ == '__main__':
    data = datagenerator(data_dir="None")
