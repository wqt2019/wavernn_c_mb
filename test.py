

import numpy as np
import torch
from utils.pqmf import PQMF
import librosa
import sys

sample_rate = 16000
def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=sample_rate)


def npy2txt():
    np.set_printoptions(threshold=sys.maxsize)
    boxes = np.load('/home/wqt/software/clion/work/wavernn_c_mb/16000/biaobei_mel/000001.npy')
    np.savetxt('/home/wqt/software/clion/work/wavernn_c_mb/16000/txt/000001.txt', boxes, fmt='%s', newline='\n')


def pqmf0():
    pqmf = PQMF()

    txt_file = '/home/wqt/software/clion/work/wavernn_c_mb/16000/result1.txt'

    f_txt = open(txt_file,'r')
    txt_ = f_txt.readlines()
    txt_len = len(txt_)
    output = np.zeros([txt_len,4])

    for i in range(txt_len):
        i_split = txt_[i].split(' ')
        output[i][0] = float(i_split[0])
        output[i][1] = float(i_split[1])
        output[i][2] = float(i_split[2])
        output[i][3] = float(i_split[3])
        aa = 0

    output = output.transpose()
    output = output[np.newaxis, :]
    output = output.astype(np.float32)
    save_path = '/home/wqt/software/clion/work/wavernn_c_mb/16000/test1.wav'
    output_pqmf = pqmf.synthesis(torch.from_numpy(output)).squeeze().numpy()
    save_wav(output_pqmf, save_path)

    return


def printVector(f, vector, name, dtype='float'):
    v = np.reshape(vector, (-1))
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return


def pqmf1():
    subbands = 4
    updown_filter = np.zeros((4, 4, 4),dtype= np.float32)
    for k in range(subbands):
        updown_filter[k, k, 0] = 1.0 *subbands
    x = np.loadtxt('x.txt',dtype= np.float32)
    x1 = np.loadtxt('x1.txt',dtype= np.float32)
    x2 = np.loadtxt('x2.txt',dtype= np.float32)
    x3 = np.loadtxt('x3.txt',dtype= np.float32)
    synthesis_filter = np.loadtxt('synthesis_filter.txt', dtype=np.float32)

    fcfile = 'synthesis_filter.c'
    fc = open(fcfile, 'w')
    printVector(fc,synthesis_filter,'synthesis_filter')
    fc.close()

    xh,xw = x.shape
    pad = 0
    ksize = 4
    taps = 31
    ow = subbands * (xw - 1) -2*pad + ksize
    x22 = np.zeros((subbands, ow + taps * 2), dtype=np.float32)

    conv1d_pad = 0
    conv1d_ksize = 63
    conv1d_stride = 1
    conv1d_ow = int((ow + taps * 2 - conv1d_ksize + 2*conv1d_pad)/conv1d_stride)+1
    x33 = np.zeros((conv1d_ow), dtype=np.float32)

    # x1 = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
    # x2 = self.pad_fn(x1)
    for i in range(xw):
        x22[0, i * 4 + taps] = x[0, i] * updown_filter[0, 0, 0]
        x22[1, i * 4 + taps] = x[1, i] * updown_filter[1, 1, 0]
        x22[2, i * 4 + taps] = x[2, i] * updown_filter[2, 2, 0]
        x22[3, i * 4 + taps] = x[3, i] * updown_filter[3, 3, 0]

    # x3 = F.conv1d(x2, self.synthesis_filter)
    for i in range(conv1d_ow):
        out0 = 0.0
        out1 = 0.0
        out2 = 0.0
        out3 = 0.0
        for j in range(conv1d_ksize):
            out0 += x22[0, i + j] * synthesis_filter[0, j]
            out1 += x22[1, i + j] * synthesis_filter[1, j]
            out2 += x22[2, i + j] * synthesis_filter[2, j]
            out3 += x22[3, i + j] * synthesis_filter[3, j]
        x33[i] = out0+out1+out2+out3

    save_wav(x33, 'x33.wav')

    return



if __name__ == '__main__':

    # npy2txt()
    pqmf0()
    # pqmf1()

