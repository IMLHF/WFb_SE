# import soundfile
import librosa
import librosa.output
import numpy as np
# from pypesq import pesq
from pystoi.stoi import stoi
import soundfile as sf
import FLAGS
import time
import os
import shutil
from mir_eval.separation import bss_eval_sources
from numpy import linalg
from utils.assess.core import calc_pesq
AMP_MAX = (2 ** (FLAGS.PARAM.AUDIO_BITS - 1) - 1)

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''


def read_audio(file):
  data, sr = sf.read(file)
  if sr != FLAGS.PARAM.FS:
    data = librosa.resample(data, sr, FLAGS.PARAM.FS, res_type='kaiser_fast')
    print('resample wav(%d to %d) :' % (sr, FLAGS.PARAM.FS), file)
    # librosa.output.write_wav(file, data, FLAGS.PARAM.FS)
  return data*AMP_MAX, FLAGS.PARAM.FS


def write_audio(file, data, sr):
  return sf.write(file, data/AMP_MAX, sr)


def repeat_to_len(wave, repeat_len):
  while len(wave) < repeat_len:
    wave = np.tile(wave, 2)
  wave = wave[0:repeat_len]
  return wave


def _mix_wav_by_SNR(waveData, noise, snr):
  As = linalg.norm(waveData)
  An = linalg.norm(noise)

  alpha = As/(An*(10**(snr/20))) if An != 0 else 0
  waveMix = (waveData+alpha*noise)/(1.0+alpha)
  return waveMix, alpha


def _mix_wav_by_randomSNR(waveData, noise):
  # S = (speech+alpha*noise)/(1+alpha)
  snr = np.random.randint(FLAGS.PARAM.MIN_SNR, FLAGS.PARAM.MAX_SNR+1)
  return _mix_wav_by_SNR(waveData, noise, snr)


def _mix_wav_randomLINEAR(waveData, noise):
  coef = np.random.random()*(FLAGS.PARAM.MAX_COEF-FLAGS.PARAM.MIN_COEF)+FLAGS.PARAM.MIN_COEF
  waveMix = (waveData+coef*noise)/(1.0+coef)
  return waveMix, coef


def cal_SDR(src_ref, src_deg):
    """Calculate Source-to-Distortion Ratio(SDR).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_deg: numpy.ndarray, [C, T], reordered by best PIT permutation
    Returns:
        SDR
    """
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_deg)
    return sdr


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [C,T]
    Returns:
        average_SDRi
    """
    src_anchor = mix
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    sum_SDRi = 0
    for i in range(len(sdr)):
      sum_SDRi += sdr[i]-sdr0[i]
      print(sdr[i], sdr0[i])
    avg_SDRi = sum_SDRi / len(sdr)
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def get_batch_pesq_improvement(x_wav,y_wav,y_wav_est,batch_num,set_name):
  '''
  inputs:
    x_wav, y_wav, y_wav_est: [batch,wave]
  return:
     mixture pesq, enhanced pesq, pesq improvement: [batch]
  '''
  # calculate PESQ improvement
  pesq_ref_cleaned_list = [calc_pesq(ref, cleaned, FLAGS.PARAM.FS)
                           for ref, cleaned in zip(y_wav, y_wav_est)]
  pesq_ref_mixed_list = [calc_pesq(ref, mixed, FLAGS.PARAM.FS)
                         for ref, mixed in zip(y_wav, x_wav)]
  pesq_ref_cleaned_vec = np.array(pesq_ref_cleaned_list)
  pesq_ref_mixed_vec = np.array(pesq_ref_mixed_list)
  pesq_imp_vec = pesq_ref_cleaned_vec - pesq_ref_mixed_vec

  if FLAGS.PARAM.GET_AUDIO_IN_TEST:
    decode_ans_file = os.path.join(FLAGS.PARAM.SAVE_DIR,'decode_'+FLAGS.PARAM.CHECK_POINT, set_name)
    if os.path.exists(decode_ans_file) and batch_num == 1:
      shutil.rmtree(decode_ans_file)
    if not os.path.exists(decode_ans_file):
      os.makedirs(decode_ans_file)
    for i, ref, cleaned, mixed, pesqi in zip(range(len(y_wav)), y_wav, y_wav_est, x_wav, pesq_imp_vec):
      write_audio(os.path.join(decode_ans_file, "%04d_%03d_ref.wav" % (batch_num, i)),
                  ref, FLAGS.PARAM.FS)
      write_audio(os.path.join(decode_ans_file, "%04d_%03d_cleaned_%.2f.wav" % (batch_num, i, pesqi)),
                  cleaned, FLAGS.PARAM.FS)
      write_audio(os.path.join(decode_ans_file, "%04d_%03d_mixed.wav" % (batch_num, i)),
                  mixed, FLAGS.PARAM.FS)

  return np.array([pesq_ref_mixed_vec, pesq_ref_cleaned_vec, pesq_imp_vec])


def get_batch_stoi_improvement(x_wav,y_wav,y_wav_est):
  # return np.array([[1,1],[2,2],[3,3]])
  '''
  inputs:
    x_wav, y_wav, y_wav_est: [batch,wave]
  return:
     mixture stoi, enhanced stoi, stoi improvement: [batch]
  '''
  # calculate STOI improvement
  stoi_ref_cleaned_list = [stoi(ref/AMP_MAX,
                                cleaned/AMP_MAX,
                                FLAGS.PARAM.FS)
                           for ref, cleaned in zip(y_wav, y_wav_est)]
  stoi_ref_mixed_list = [stoi(ref/AMP_MAX,
                              mixed/AMP_MAX,
                              FLAGS.PARAM.FS)
                         for ref, mixed in zip(y_wav, x_wav)]
  stoi_ref_cleaned_vec = np.array(stoi_ref_cleaned_list)
  stoi_ref_mixed_vec = np.array(stoi_ref_mixed_list)
  stoi_imp_vec = stoi_ref_cleaned_vec - stoi_ref_mixed_vec
  return np.array([stoi_ref_mixed_vec, stoi_ref_cleaned_vec, stoi_imp_vec])


def get_batch_sdr_improvement(x_wav,y_wav,y_wav_est):
  # calculate STOI improvement
  sdr_ref_cleaned_list = [cal_SDR(ref/AMP_MAX,
                                  cleaned/AMP_MAX)[0]
                          for ref, cleaned in zip(y_wav, y_wav_est)]
  sdr_ref_mixed_list = [cal_SDR(ref/AMP_MAX,
                                mixed/AMP_MAX)[0]
                        for ref, mixed in zip(y_wav, x_wav)]
  # print(np.shape(sdr_ref_cleaned_list))
  sdr_ref_cleaned_vec = np.array(sdr_ref_cleaned_list)
  sdr_ref_mixed_vec = np.array(sdr_ref_mixed_list)
  sdr_imp_vec = sdr_ref_cleaned_vec - sdr_ref_mixed_vec
  return np.array([sdr_ref_mixed_vec, sdr_ref_cleaned_vec, sdr_imp_vec])


def gen_mixed_wav(ref_dir, noise_dir, mixed_dir, snr=0, force=False):
  ref_list = os.listdir(ref_dir)
  ref_list = [os.path.join(ref_dir,ref_file) for ref_file in ref_list]
  ref_list.sort()

  noise_list = os.listdir(noise_dir)
  noise_list = [os.path.join(noise_dir,noise_file) for noise_file in noise_list]
  noise_list.sort()

  gen_mixed = False
  if not os.path.exists(mixed_dir):
    os.makedirs(mixed_dir)
    gen_mixed = True
  elif force is True:
    shutil.rmtree(mixed_dir)
    os.makedirs(mixed_dir)
    gen_mixed = True

  ref_list_full = []
  mixed_list = []
  for ref_file in ref_list:
    if gen_mixed:
      ref_wave, ref_sr = read_audio(ref_file)
      ref_len = len(ref_wave)
    ref_name = ref_file[ref_file.rfind('/')+1:ref_file.rfind('.')]
    for noise_file in noise_list:
      noise_name = noise_file[noise_file.rfind('/')+1:noise_file.rfind('.')]
      mixed_file = os.path.join(mixed_dir, ref_name+'_MIX_'+noise_name+'.wav')
      if gen_mixed:
        noise_wave, noise_sr = read_audio(noise_file)
        noise_wave = repeat_to_len(noise_wave, ref_len)
        mixed_wave, noise_alpha = _mix_wav_by_SNR(ref_wave, noise_wave, snr)
        write_audio(mixed_file,
                    mixed_wave, ref_sr)
        if noise_sr != ref_sr:
          print('noise sr is not equal to ref sr.')
          exit(-1)
        print(ref_name+'_MIX_'+noise_name)
      ref_list_full.append(ref_file)
      mixed_list.append(mixed_file)

  return ref_list_full, mixed_list
