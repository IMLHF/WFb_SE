import tensorflow as tf
import numpy as np
import sys
import os
import gc
from utils import spectrum_tool
from utils import audio_tool
from pypesq import pesq
from models.lstm_SE import SE_MODEL
from FLAGS import PARAM
from dataManager.mixed_aishell_8k_tfrecord_io import generate_tfrecord, get_batch_use_tfdata

def _build_model_use_tfdata(test_set_tfrecords_dir, ckpt_dir):
  '''
  test_set_tfrecords_dir: '/xxx/xxx/*.tfrecords'
  '''
  g = tf.Graph()
  with g.as_default():
    # region TFRecord+DataSet
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch, y_batch, Xtheta_batch, Ytheta_batch, lengths_batch, iter_train = get_batch_use_tfdata(
          test_set_tfrecords_dir,
          get_theta=True)

    with tf.name_scope('model'):
      test_model = SE_MODEL(x_batch,
                            lengths_batch,
                            y_batch,
                            Xtheta_batch,
                            Ytheta_batch,
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
      test_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
  g.finalize()
  return sess, test_model


def get_PESQ_STOI_SDR(test_set_tfrecords_dir, ckpt_dir):
  '''
  x_mag : magnitude spectrum of mixed audio.
  x_theta : angle of mixed audio's complex spectrum.
  y_ : clean(label) audio's ...
  y_xx_est " estimate audio's ...
  '''
  sess, model = _build_model_use_tfdata(test_set_tfrecords_dir, ckpt_dir)
  mask, x_mag, x_theta, y_mag, y_theta, y_mag_est = sess.run([model.mask,
                                                              model.inputs,
                                                              model.x_theta,
                                                              model.labels,
                                                              model.y_theta,
                                                              model.cleaned])
  x_wav = spectrum_tool.librosa_istft(x_mag*np.exp(1j*x_theta), PARAM.NFFT, PARAM.OVERLAP)
  y_wav = spectrum_tool.librosa_istft(y_mag*np.exp(1j*y_theta), PARAM.NFFT, PARAM.OVERLAP)
  if PARAM.RESTORE_PHASE == 'MIXED':
    y_spec_est = y_mag_est*np.exp(1j*x_theta)
    y_wav_est = spectrum_tool.librosa_istft(y_spec_est, PARAM.NFFT, PARAM.OVERLAP)
  elif PARAM.RESTORE_PHASE =='GRIFFIN_LIM':
    y_wav_est = spectrum_tool.griffin_lim(y_mag_est,
                                          PARAM.NFFT,
                                          PARAM.OVERLAP,
                                          PARAM.GRIFFIN_ITERNUM,
                                          x_wav)
  pesq_ans = audio_tool.get_batch_pesq_improvement(x_wav, y_wav, y_wav_est)
  stoi_ans = audio_tool.get_batch_stoi_improvement(x_wav, y_wav, y_wav_est)
  sdr_ans = audio_tool.get_batch_sdr_improvement(x_wav, y_wav, y_wav_est)
  return pesq_ans, stoi_ans, sdr_ans

def test_CC_and_OC():
  ckpt_dir = PARAM.CHECK_POINT
  _, _, testcc_tfrecords_dir, testoc_tfrecords_dir = generate_tfrecord(
      gen=PARAM.GENERATE_TFRECORD)
  pesq_ans, stoi_ans, sdr_ans = get_PESQ_STOI_SDR(testcc_tfrecords_dir, ckpt_dir)
