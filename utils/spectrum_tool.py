import numpy as np
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
# plt.switch_backend('agg')
# ffmpeg -i 20180829_191732_mono.wav -ar 16000 -ac 1 20180829_191732_mono16k.wav


def picture_spec(spec, name):
  '''
  spec:[time,fft_dot]
  '''
  spec_t = spec.T
  plt.figure(figsize=(12, 5))
  plt.pcolormesh(spec_t)
  plt.title('STFT Magnitude')
  plt.xlabel('Time')
  plt.ylabel('Frequency')
  plt.savefig(name+".jpg")
  print("write pic "+name)
  plt.close()


def picture_wave(wave_t, name, framerate):
  nframes = np.shape(wave_t)[0]
  _time = np.arange(0, nframes)*(1.0 / framerate)
  plt.plot(_time, wave_t)
  plt.xlabel("Time")
  plt.ylabel("Amplitude")
  plt.title("Single channel wave")
  plt.grid(True)
  plt.savefig(name+".jpg")
  # plt.show()
  print("write pic "+name)
  plt.close()


def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  '''
  signal: [wave_len]
  return mag_spec:[time,fft_dot]
  '''
  signal = np.array(signal, dtype=np.float32)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T  # tmp.T:[time, fft_dot]


def phase_spectrum_librosa_stft(signal, NFFT, overlap):
  '''
  signal: [wave_len]
  return theta:[time, angle_vector]
  '''
  signal = np.array(signal, dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.angle(tmp)
  return tmp.T  # tmp.T:[time, angle_vector]


def librosa_istft(magnitude_complex, NFFT, overlap):
  '''
  magnitude_complex:[time,frequence]
  return: [wave_len]
  '''
  tmp = librosa.core.istft(magnitude_complex.T,
                           win_length=NFFT,
                           hop_length=NFFT-overlap,
                           window=scipy.signal.windows.hann)
  return tmp # wave


def griffin_lim(spec, NFFT, overlap, max_iter, mixed_wav):
  '''
  spec:[time,fft_dot]
  '''
  # y = np.random.random(np.shape(librosa_istft(spec,
  #                                             NFFT=NFFT,
  #                                             overlap=overlap,)))
  y = mixed_wav
  for i in range(max_iter-1):
    stft_matrix = librosa.core.stft(y,
                                    n_fft=NFFT,
                                    hop_length=NFFT-overlap,
                                    window=scipy.signal.windows.hann)
    stft_matrix = stft_matrix.T
    # stft_matrix = spec * (stft_matrix / np.maximum(np.abs(stft_matrix),1e-10))
    stft_matrix = spec * np.exp(1j*np.angle(stft_matrix))
    y = librosa.core.istft(stft_matrix.T,
                           win_length=NFFT,
                           hop_length=NFFT-overlap,
                           window=scipy.signal.windows.hann)
  return y
