from utils import audio_tool,spectrum_tool,tf_tool
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
if __name__ == "__main__":
  noise_list = os.listdir('exp/real_test_fair/7_noise')
  noise_list = [os.path.join('exp/real_test_fair/7_noise',_dir) for _dir in noise_list]

  s_dir = 'exp/real_test_fair/863_min_16k/refer_wav/M1014.wav'


  for n_dir in noise_list:
    s_data,sr = audio_tool.read_audio(s_dir)
    n_data,sr = audio_tool.read_audio(n_dir)
    n_data = audio_tool.repeat_to_len(n_data,len(s_data))
    mix_data,_ = audio_tool._mix_wav_by_SNR(s_data,n_data,0)
    noise_name = n_dir[n_dir.rfind('/')+1:n_dir.rfind('.')]
    audio_tool.write_audio(os.path.join('exp/real_test_fair/863_min_16k/mixed_wav', 'M1014_MIX_'+noise_name+'.wav'),
                           mix_data, sr)
