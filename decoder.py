import tensorflow as tf
import numpy as np
import sys
import utils
import os
import gc
# from pypesq import pesq
from utils import spectrum_tool
from utils import audio_tool
import utils.audio_tool
from FLAGS import PARAM
import shutil
import time
import utils.assess.core as pesqexe
from pystoi import stoi


def build_session(ckpt_dir,batch_size,finalizeG=True):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch = tf.placeholder(tf.float32,shape=[batch_size,None,PARAM.INPUT_SIZE],name='x_batch')
        lengths_batch = tf.placeholder(tf.int32,shape=[batch_size],name='lengths_batch')
        y_batch = tf.placeholder(tf.float32,shape=[batch_size,None,PARAM.INPUT_SIZE],name='y_batch')
        x_theta = tf.placeholder(tf.float32,shape=[batch_size,None,PARAM.INPUT_SIZE],name='x_theta')
        y_theta = tf.placeholder(tf.float32,shape=[batch_size,None,PARAM.INPUT_SIZE],name='y_theta')
    with tf.name_scope('model'):
      model = PARAM.SE_MODEL(x_batch,
                             lengths_batch,
                             y_batch,
                             x_theta,
                             y_theta,
                             behavior=PARAM.SE_MODEL.infer)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(
        os.path.join(PARAM.SAVE_DIR, ckpt_dir))
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
  if finalizeG:
    g.finalize()
    return sess,model
  return sess,model,g


def decode_one_wav(sess, model, wavedata):
  x_spec_t = spectrum_tool.magnitude_spectrum_librosa_stft(wavedata,
                                                           PARAM.NFFT,
                                                           PARAM.OVERLAP)
  length = np.shape(x_spec_t)[0]
  x_spec = np.array([x_spec_t], dtype=np.float32)
  lengths = np.array([length], dtype=np.int32)
  y_mag_estimation, mask, x_mag, norm_x_mag, norm_logmag = sess.run(
      [model.y_mag_estimation, model.mask,
       model._x_mag_spec, model._norm_x_mag_spec, model._norm_x_logmag_spec],
      feed_dict={
          model.x_mag: x_spec,
          model.lengths: lengths,
      })

  y_mag_estimation = np.array(y_mag_estimation[0])
  mask = np.array(mask[0])
  # print(np.shape(y_mag_estimation), np.shape(mask))
  if PARAM.RESTORE_PHASE == 'MIXED':
    y_mag_estimation = y_mag_estimation*np.exp(1j*spectrum_tool.phase_spectrum_librosa_stft(wavedata,
                                                                                            PARAM.NFFT,
                                                                                            PARAM.OVERLAP))
    reY = spectrum_tool.librosa_istft(y_mag_estimation, PARAM.NFFT, PARAM.OVERLAP)
  elif PARAM.RESTORE_PHASE == 'GRIFFIN_LIM':
    reY = spectrum_tool.griffin_lim(y_mag_estimation,
                                    PARAM.NFFT,
                                    PARAM.OVERLAP,
                                    PARAM.GRIFFIN_ITERNUM,
                                    wavedata)

  # print(np.shape(mask), np.max(mask), np.min(mask))
  # print(np.shape(x_mag), np.max(x_mag), np.min(x_mag))
  # print(np.shape(norm_x_mag), np.max(norm_x_mag), np.min(norm_x_mag))
  # print(np.shape(norm_logmag), np.max(norm_logmag), np.min(norm_logmag))
  # print(np.shape(y_mag_estimation), np.max(y_mag_estimation), np.min(y_mag_estimation))
  # spectrum_tool.picture_spec(mask[0],"233")
  return reY, mask

def decode_and_getMeature(decode_file_list, ref_list, sess, model, decode_ans_file, save_audio, ans_file):
  if os.path.exists(os.path.join(decode_ans_file,ans_file)):
    os.remove(os.path.join(decode_ans_file,ans_file))
  pesq_raw_sum = 0
  pesq_en_sum = 0
  stoi_raw_sum = 0
  stoi_en_sum = 0
  sdr_raw_sum = 0
  sdr_en_sum = 0
  for i, mixed_dir in enumerate(decode_file_list):
    print('\n',i+1,mixed_dir)
    waveData, sr = utils.audio_tool.read_audio(mixed_dir)
    reY, mask = decode_one_wav(sess,model,waveData)
    abs_max = (2 ** (PARAM.AUDIO_BITS - 1) - 1)
    reY = np.where(reY > abs_max, abs_max, reY)
    reY = np.where(reY < -abs_max, -abs_max, reY)
    file_name = mixed_dir[mixed_dir.rfind('/')+1:mixed_dir.rfind('.')]
    if save_audio:
      utils.audio_tool.write_audio(os.path.join(decode_ans_file,
                                                (ckpt+'_%03d_' % (i+1))+mixed_dir[mixed_dir.rfind('/')+1:]),
                                   reY,
                                   sr)
      spectrum_tool.picture_spec(mask,
                                 os.path.join(decode_ans_file,
                                              (ckpt+'_%03d_' % (i+1))+file_name))

    if i<len(ref_list):
      ref, sr = utils.audio_tool.read_audio(ref_list[i])
      print(' refer: ',ref_list[i])
      len_small = min(len(ref),len(waveData),len(reY))
      ref = np.array(ref[:len_small])
      waveData = np.array(waveData[:len_small])
      reY = np.array(reY[:len_small])
      # sdr
      sdr_raw = audio_tool.cal_SDR(np.array([ref]),
                                   np.array([waveData]))
      sdr_en = audio_tool.cal_SDR(np.array([ref]),
                                  np.array(reY))
      sdr_raw_sum += sdr_raw
      sdr_en_sum += sdr_en
      # pesq
      # pesq_raw = pesq(ref,waveData,sr)
      # pesq_en = pesq(ref,reY,sr)
      pesq_raw = pesqexe.calc_pesq(ref,waveData,sr)
      pesq_en = pesqexe.calc_pesq(ref,reY,sr)
      pesq_raw_sum += pesq_raw
      pesq_en_sum += pesq_en
      # stoi
      stoi_raw = stoi.stoi(ref,waveData,sr)
      stoi_en = stoi.stoi(ref,reY,sr)
      stoi_raw_sum += stoi_raw
      stoi_en_sum += stoi_en
      print("SR = %d" % sr)
      print("PESQ_raw: %.3f, PESQ_en: %.3f, PESQimp: %.3f. " % (pesq_raw,pesq_en,pesq_en-pesq_raw))
      print("SDR_raw: %.3f, SDR_en: %.3f, SDRimp: %.3f. " % (sdr_raw,sdr_en,sdr_en-sdr_raw))
      print("STOI_raw: %.3f, STOI_en: %.3f, STOIimp: %.3f. " % (stoi_raw,stoi_en,stoi_en-stoi_raw))
      sys.stdout.flush()
      with open(os.path.join(decode_ans_file,ans_file),'a+') as f:
        f.write(file_name+'\r\n')
        f.write("    |-PESQ_raw: %.3f, PESQ_en: %.3f, PESQimp: %.3f. \r\n" % (pesq_raw,pesq_en,pesq_en-pesq_raw))
        f.write("    |-SDR_raw: %.3f, SDR_en: %.3f, SDRimp: %.3f. \r\n" % (sdr_raw,sdr_en,sdr_en-sdr_raw))
        f.write("    |-STOI_raw: %.3f, STOI_en: %.3f, STOIimp: %.3f. \r\n" % (stoi_raw,stoi_en,stoi_en-stoi_raw))

  len_list = len(ref_list)
  with open(os.path.join(decode_ans_file,ans_file),'a+') as f:
    f.write('PESQ_raw:%.3f, PESQ_en:%.3f, PESQi_avg:%.3f. \r\n' % (pesq_raw_sum/len_list, pesq_en_sum/len_list, (pesq_en_sum-pesq_raw_sum)/len_list))
    f.write('SDR_raw:%.3f, SDR_en:%.3f, SDRi_avg:%.3f. \r\n' % (sdr_raw_sum/len_list, sdr_en_sum/len_list, (sdr_en_sum-sdr_raw_sum)/len_list))
    f.write('STOI_raw:%.3f, STOI_en:%.3f, STOIi_avg:%.3f. \r\n' % (stoi_raw_sum/len_list, stoi_en_sum/len_list, (stoi_en_sum-stoi_raw_sum)/len_list))
  print('\n\n\n-----------------------------------------')
  print('PESQ_raw:%.3f, PESQ_en:%.3f, PESQi_avg:%.3f. \r\n' % (pesq_raw_sum/len_list, pesq_en_sum/len_list, (pesq_en_sum-pesq_raw_sum)/len_list))
  print('SDR_raw:%.3f, SDR_en:%.3f, SDRi_avg:%.3f. \r\n' % (sdr_raw_sum/len_list, sdr_en_sum/len_list, (sdr_en_sum-sdr_raw_sum)/len_list))
  print('STOI_raw:%.3f, STOI_en:%.3f, STOIi_avg:%.3f. \r\n' % (stoi_raw_sum/len_list, stoi_en_sum/len_list, (stoi_en_sum-stoi_raw_sum)/len_list))
  sys.stdout.flush()


if __name__=='__main__':
  ckpt= PARAM.CHECK_POINT # don't forget to change FLAGS.PARAM
  decode_ans_file = os.path.join(PARAM.SAVE_DIR,'decode_'+ckpt)
  if not os.path.exists(decode_ans_file):
    os.makedirs(decode_ans_file)
  sess, model = build_session(ckpt, 1)

  decode_file_list = [
      'exp/rnn_speech_enhancement/s_2_00_MIX_1_clapping_8k.wav',
      'exp/rnn_speech_enhancement/s_8_01_MIX_4_rainning_8k.wav',
      'exp/rnn_speech_enhancement/s_8_21_MIX_3_factory_8k.wav',
      'exp/rnn_speech_enhancement/s_2_00_8k_raw.wav',
      'exp/rnn_speech_enhancement/s_8_01_8k_raw.wav',
      'exp/rnn_speech_enhancement/s_8_21_8k_raw.wav',
      'exp/rnn_speech_enhancement/speech1_8k.wav',
      'exp/rnn_speech_enhancement/speech5_8k.wav',
      'exp/rnn_speech_enhancement/speech6_8k.wav',
      'exp/rnn_speech_enhancement/speech7_8k.wav',
      'exp/real_test_fair/863_min/mixed_wav/863_1_8k_MIX_1_airplane.wav',
      # 'exp/rnn_speech_enhancement/decode_nnet_C001_3/nnet_C001_3_007_speech7_8k.wav'
  ]


  if len(sys.argv)<=1:
    for i, mixed_dir in enumerate(decode_file_list):
      print(i+1,mixed_dir)
      waveData, sr = utils.audio_tool.read_audio(mixed_dir)
      reY, mask = decode_one_wav(sess,model,waveData)
      print(np.max(reY))
      abs_max = (2 ** (PARAM.AUDIO_BITS - 1) - 1)
      reY = np.where(reY>abs_max,abs_max,reY)
      reY = np.where(reY<-abs_max,-abs_max,reY)
      utils.audio_tool.write_audio(os.path.join(decode_ans_file,
                                                (ckpt+'_%03d_' % (i+1))+mixed_dir[mixed_dir.rfind('/')+1:]),
                                   reY,
                                   sr)
      file_name = mixed_dir[mixed_dir.rfind('/')+1:mixed_dir.rfind('.')]
      spectrum_tool.picture_spec(mask,
                                 os.path.join(decode_ans_file,
                                              (ckpt+'_%03d_' % (i+1))+file_name))
  elif int(sys.argv[1])==0:
    ref_list = [
      'exp/rnn_speech_enhancement/2_00_8k.wav',
      'exp/rnn_speech_enhancement/8_01_8k.wav',
      'exp/rnn_speech_enhancement/8_21_8k.wav',
      'exp/rnn_speech_enhancement/2_00_8k.wav',
      'exp/rnn_speech_enhancement/8_01_8k.wav',
      'exp/rnn_speech_enhancement/8_21_8k.wav',
    ]
    decode_and_getMeature(decode_file_list, ref_list, sess, model, decode_ans_file, True, 'sdr.txt')
  elif int(sys.argv[1])==1:
    start_time = time.time()
    mixed_dirs = os.path.join('exp','real_test_fair','ITU_T_Test', 'mixed_wav')
    # mixed_dirs = os.path.join('exp','real_test_fair','863_min', 'mixed_wav')
    mixed_list = os.listdir(mixed_dirs)
    mixed_list = [os.path.join(mixed_dirs,mixed_file) for mixed_file in mixed_list]
    mixed_list.sort()

    refer_dirs = os.path.join('exp','real_test_fair','ITU_T_Test', 'refer_wav')
    # refer_dirs = os.path.join('exp','real_test_fair','863_min', 'refer_wav')
    refer_list_single = os.listdir(refer_dirs)
    refer_list_single = [os.path.join(refer_dirs,refer_file) for refer_file in refer_list_single]
    refer_list_single.sort()
    refer_list = []
    for wav_dir in refer_list_single:
      for i in range(7):
        refer_list.append(wav_dir)
    decode_and_getMeature(mixed_list, refer_list, sess, model, decode_ans_file, False, 'real_testITU_T.txt')
    # decode_and_getMeature(mixed_list, refer_list, sess, model, decode_ans_file, True, 'real_test863min.txt')
    print(ckpt)
    print("Cost time : %dS" % (time.time()-start_time))
