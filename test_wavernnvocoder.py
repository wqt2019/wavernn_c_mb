#-*- coding: utf-8 -*-

import sys,time
import glob,os
from scipy.io.wavfile import write
from scipy.io import wavfile
import librosa

sys.path.insert(0,'./cmake-build-debug/')
import WaveRNNVocoder
import numpy as np
import math
import soundfile as sf


sr = 16000 
n_mel = 80
model_file = './16000/biaobei/model_mb_std.bin'
mel_path = './16000/biaobei_std_mel_test/*.npy'
save_path = './16000/'
name = 'biaobei_mb_batch_'

if not os.path.exists(save_path):
    os.mkdir(save_path)

vocoder =WaveRNNVocoder.Vocoder()
vocoder.loadWeights(model_file)

filelist = glob.glob(mel_path)
filelist.sort()


i = 200
for fname in filelist:
    print(fname)
    mel = np.load(fname)
    #mel = (mel+4)/8
    if(mel.shape[0] !=n_mel):
        mel = np.transpose(mel, (1, 0))
    t1 = time.time()
    print('start')
    wav = vocoder.melToWav(mel)

    spend_time = time.time() - t1
    #wav = np.int16(wav/np.max(np.abs(wav)) * 32767)

    # librosa.output.write_wav(save_path + str(i) + '.wav', wav.astype(np.float32), sr=sr)
    #wavfile.write(save_path + str(i) + '.wav', sr, wav)
    sf.write(save_path + name + str(i) + '.wav', wav, sr, 'PCM_16')
	
    print('\npoints/s:',len(wav)/(spend_time))
    print('generate wav duration:',len(wav)/(sr))
    print('generate spend time:',spend_time)
    print('---------\n')
    i += 1

print('\ndone')
