import tensorflow as tf
import numpy as np
import sys
import utils
import os
from models.lstm_SE import SE_MODEL
import gc
from utils import spectrum_tool
from FLAGS import PARAM
from utils import audio_tool


def build_session(ckpt_dir):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch = tf.placeholder(tf.float32,shape=[1,None,PARAM.INPUT_SIZE],name='x_batch')
        lengths_batch = tf.placeholder(tf.int32,shape=[1],name='lengths_batch')
    with tf.name_scope('model'):
      model = SE_MODEL(x_batch,
                       lengths_batch,
                       infer=True)

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
  g.finalize()
  return sess,model


def decode_one_wav(sess, model, wavedata):
  x_spec_t = spectrum_tool.magnitude_spectrum_librosa_stft(wavedata,
                                                           PARAM.NFFT,
                                                           PARAM.OVERLAP)
  length = np.shape(x_spec_t)[0]
  x_spec = np.array([x_spec_t], dtype=np.float32)
  lengths = np.array([length], dtype=np.int32)
  cleaned, mask, x_mag, norm_x_mag, norm_logmag = sess.run(
      [model.cleaned, model.mask,
       model._x_mag_spec, model._norm_x_mag_spec, model._norm_x_logmag_spec],
      feed_dict={
        model.x_mag: x_spec,
        model.lengths: lengths,
      })

  cleaned = np.array(cleaned[0])
  mask = np.array(mask[0])
  print(np.shape(cleaned),np.shape(mask))
  if PARAM.RESTORE_PHASE == 'MIXED':
    cleaned = cleaned*np.exp(1j*spectrum_tool.phase_spectrum_librosa_stft(wavedata,
                                                                          PARAM.NFFT,
                                                                          PARAM.OVERLAP))
    reY = spectrum_tool.librosa_istft(cleaned, PARAM.NFFT, PARAM.OVERLAP)
  elif PARAM.RESTORE_PHASE =='GRIFFIN_LIM':
    reY = spectrum_tool.griffin_lim(cleaned,
                                    PARAM.NFFT,
                                    PARAM.OVERLAP,
                                    PARAM.GRIFFIN_ITERNUM,
                                    wavedata)

  print(np.shape(mask), np.max(mask), np.min(mask))
  print(np.shape(x_mag), np.max(x_mag), np.min(x_mag))
  print(np.shape(norm_x_mag), np.max(norm_x_mag), np.min(norm_x_mag))
  print(np.shape(norm_logmag), np.max(norm_logmag), np.min(norm_logmag))
  # spectrum_tool.picture_spec(mask[0],"233")
  return reY, mask

if __name__=='__main__':
  ckpt= PARAM.CHECK_POINT # don't forget to change FLAGS.PARAM
  decode_ans_file = os.path.join(PARAM.SAVE_DIR,'decode_'+ckpt)
  if not os.path.exists(decode_ans_file):
    os.makedirs(decode_ans_file)
  sess, model = build_session(ckpt)

  decode_file_list = ['../IRM_Speech_Enhancement/exp/data_for_ac/mixed_wav_c11_50_snr_0/2_00_MIX_1_clapping.wav',
                      '../IRM_Speech_Enhancement/_decode_index/speech5_16k.wav',
                      '../IRM_Speech_Enhancement/_decode_index/speech0_16k.wav',]

  for i, mixed_dir in enumerate(decode_file_list):
    print(i+1,mixed_dir)
    waveData, sr = audio_tool.read_audio(mixed_dir)
    reY, mask = decode_one_wav(sess,model,waveData)
    utils.audio_tool.write_audio(os.path.join(decode_ans_file,
                                              ('%3d_' % (i+1))+mixed_dir[mixed_dir.rfind('/')+1:]),
                                 reY,
                                 sr)
    file_name = mixed_dir[mixed_dir.rfind('/')+1:mixed_dir.rfind('.')]
    spectrum_tool.picture_spec(mask,
                               os.path.join(decode_ans_file,
                                            ('%3d_' % (i+1))+file_name))
