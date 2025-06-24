import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import torch.nn as nn
from torch.autograd import Variable
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from models_DNN import DnCNN
import numpy as np
import pywt
import astropy.io.fits as pyfits
import sys
import datetime
import ephem
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
import torch

secperday = 3600 * 24

def readfits(filename):
    hdulist = pyfits.open(filename)
    hdu0 = hdulist[0]
    hdu1 = hdulist[1]
    data1 = hdu1.data['data']
    header1 = hdu1.header
    fchannel = hdulist['SUBINT'].data[0]['DAT_FREQ']
    obsfreq = hdu0.header['OBSFREQ']
    obsnchan = hdu0.header['OBSNCHAN']
    obsbw = hdu0.header['OBSBW']
    fch1 = fchannel[0]
    fmin = obsfreq - obsbw/2.
    fmax = obsfreq + obsbw/2.
    tsamp = hdu1.header['TBIN']
    nsubint = hdu1.header['NAXIS2']
    samppersubint = int(hdu1.header['NSBLK'])
    ra = hdu0.header['RA']
    dec = hdu0.header['DEC']
    subintoffset = hdu1.header['NSUBOFFS']
    tstart = "%.18f" % (Decimal(hdu0.header['STT_IMJD']) + Decimal(hdu0.header['STT_SMJD'] + tsamp * samppersubint * subintoffset )/secperday )
    jd=float(tstart)+2400000.5
    date=ephem.julian_date('1899/12/31 12:00:00')
    djd=jd-date
    str1=ephem.Date(djd).datetime()
    str2=str1.strftime('%Y-%m-%d %H:%M:%S')
    a,b,c,d,e = data1.shape
    if c > 1:
         data = data1[:,:,1,:,:].squeeze().reshape((-1,d))
    else:
         data = data1.squeeze().reshape((-1,d))
    return data


def wavelet_denoise_2d(Data):
    coeffs = pywt.wavedec2(Data, 'db8', level=3)
    coeffs = list(coeffs)
    for i in range(len(coeffs)):
        coeffs[i] = list(coeffs[i])
    return coeffs

def inverse_wavelet_transform(denoised_coeffs, coeffs):
    coeffs[0] = denoised_coeffs[0]
    for i in range(1,len(denoised_coeffs)):
        coeffs[i][0] = denoised_coeffs[i]
    denoised_data = pywt.waverec2(coeffs, 'db8')
    return denoised_data

def wavelet_denoise_2dd(data, model0, model10, model20, device):
        scaler = RobustScaler()
        data = data.astype(np.float32)
        level = 3
        co = pywt.wavedec2(data, 'db8', level=level)
        denoised_coeffs = []
        # 处理第一个系数
        scaler.fit(co[0])
        data1 = scaler.transform(co[0])
        data1 = torch.tensor(data1).unsqueeze(0).unsqueeze(0).to(device)  # 使用.to(device)确保在正确的设备上
        data1 = model0(data1)
        output0 = data1.squeeze(0).squeeze(0).cpu().detach().numpy()  # 处理完成后移回CPU
        co[0] = (output0 * scaler.scale_) + scaler.center_
        denoised_coeffs.append(co[0])

        # 处理其他系数
        for i in range(1, level):
            scaler.fit(co[i][0])
            data2 = scaler.transform(co[i][0])
            data2 = torch.tensor(data2).unsqueeze(0).unsqueeze(0).to(device)  # 使用.to(device)确保在正确的设备上
            if i == 1:
                data2 = model10(data2)
            elif i == 2:
                data2 = model20(data2)
            output10 = data2.squeeze(0).squeeze(0).cpu().detach().numpy()  # 处理完成后移回CPU
            output10 = (output10 * scaler.scale_) + scaler.center_
            denoised_coeffs.append(output10)

        coeffs = wavelet_denoise_2d(data)
        data = inverse_wavelet_transform(denoised_coeffs, coeffs)

        return data

def main(data='None'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model0 = DnCNN().to(device)
        model10 = DnCNN().to(device)
        model20 = DnCNN().to(device)
        model0 = nn.DataParallel(model0).to(device)
        model10 = nn.DataParallel(model10).to(device)
        model20 = nn.DataParallel(model20).to(device)

        model_name = 'model_010.pth'
        model0.load_state_dict(
            torch.load(os.path.join('/home/Youzy/panyr/DNCNN/pytorch tensor/models7/coeffs0', model_name),
                       map_location=device))
        model10.load_state_dict(
            torch.load(os.path.join('/home/Youzy/panyr/DNCNN/pytorch tensor/models7/coeffs10', model_name),
                       map_location=device))
        model20.load_state_dict(
            torch.load(os.path.join('/home/Youzy/panyr/DNCNN/pytorch tensor/models7/coeffs20', model_name),
                       map_location=device))
        model0.eval()
        model10.eval()
        model20.eval()
        with torch.no_grad():
            chunk_size = 4096
            num_chunks = data.shape[1] // chunk_size
            processed_data = []
            for i in range(num_chunks):
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size
                current_data = data[:, start_index:end_index]
                processed_data.append(
                    wavelet_denoise_2dd(current_data, model0=model0, model10=model10, model20=model20, device=device))
                torch.cuda.empty_cache()
                del current_data
            clean_data = np.concatenate(processed_data, axis=1)
            return clean_data

if __name__ == "__main__":
      main(data='None')
