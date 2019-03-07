# import soundfile
import librosa
import librosa.output
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import FLAGS

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''


def read_audio(file):
  data, sr = librosa.load(file)
  if sr != FLAGS.PARAM.FS:
    data = librosa.resample(data, sr, FLAGS.PARAM.FS, res_type='kaiser_fast')
  # librosa.output.write_wav(file, data, FLAGS.PARAM.FS)
  return data*32767, FLAGS.PARAM.FS


def write_audio(file, data, sr):
  data /= 32767
  return librosa.output.write_wav(file, data, sr)


def get_batch_pesq_improvement(x_wav,y_wav,y_wav_est):
  '''
  inputs:
    x_wav, y_wav, y_wav_est: [batch,wave]
  return:
     mixture pesq, enhanced pesq, pesq improvement: [batch]
  '''
  # calculate PESQ improvement
  pesq_ref_cleaned_list = [pesq(ref, cleaned, FLAGS.PARAM.FS)
                           for ref, cleaned in zip(y_wav, y_wav_est)]
  pesq_ref_mixed_list = [pesq(ref, mixed, FLAGS.PARAM.FS)
                         for ref, mixed in zip(y_wav, x_wav)]
  pesq_ref_cleaned_vec = np.array(pesq_ref_cleaned_list)
  pesq_ref_mixed_vec = np.array(pesq_ref_mixed_list)
  pesq_imp_vec = pesq_ref_cleaned_vec - pesq_ref_mixed_vec
  return np.array([pesq_ref_mixed_vec, pesq_ref_cleaned_vec, pesq_imp_vec])


def get_batch_stoi_improvement(x_wav,y_wav,y_wav_est):
  '''
  inputs:
    x_wav, y_wav, y_wav_est: [batch,wave]
  return:
     mixture stoi, enhanced stoi, stoi improvement: [batch]
  '''
  # calculate STOI improvement
  stoi_ref_cleaned_list = [stoi(ref, cleaned, FLAGS.PARAM.FS)
                           for ref, cleaned in zip(y_wav, y_wav_est)]
  stoi_ref_mixed_list = [stoi(ref, mixed, FLAGS.PARAM.FS)
                         for ref, mixed in zip(y_wav, x_wav)]
  stoi_ref_cleaned_vec = np.array(stoi_ref_cleaned_list)
  stoi_ref_mixed_vec = np.array(stoi_ref_mixed_list)
  stoi_imp_vec = stoi_ref_cleaned_vec - stoi_ref_mixed_vec
  return np.array([stoi_ref_mixed_vec, stoi_ref_cleaned_vec, stoi_imp_vec])


def get_batch_sdr_improvement(x_wav,y_wav,y_wav_est):
  return np.array([0])
