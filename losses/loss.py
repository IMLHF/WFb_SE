import tensorflow as tf
import numpy as np
import FLAGS
from utils import tf_tool

def related_reduce_sum_frame_batchsize_MSE(y1,y2):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow((y1-y2)/tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12), 2), 1), 1))
  return cost


def logbias_de_reduce_sum_frame_batchsize_MSE(y1,y2,logbias):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(y1-y2, 2), -1), -1)+tf.pow(logbias/5000,2))
  return cost


def reduce_sum_frame_batchsize_MSE_EmphasizeLowerValue(y1, y2, pow_coef):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2), pow_coef), 1), 1))
  return cost

def reduce_sum_frame_batchsize_MSE(y1, y2):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(y1-y2, 2), 1), 1))
  return cost

def fair_reduce_sum_frame_batchsize_MSE(y1, y2):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2)+1, 2)-1, 1), 1))
  return cost

def reduce_mean_MSE(y1, y2):
  cost = tf.reduce_mean(tf.pow(y1-y2, 2))
  return cost

def balanced_MFCC_AND_SPEC_MSE(y1,y2,spec_est,spec_label):
  '''
  spec_est:
    dim: [batch,time,frequence]
  spec_label:
    dim: [batch,time,frequence]
  '''
  spec_loss = reduce_sum_frame_batchsize_MSE(y1, y2)
  mfccs_est = tf_tool.mfccs_form_realStft(spec_est, FLAGS.PARAM.FS, 20, 13)
  mfccs_label = tf_tool.mfccs_form_realStft(spec_label, FLAGS.PARAM.FS, 20, 13)
  balance_coef = FLAGS.PARAM.MFCC_BLANCE_COEF
  mfcc_loss = reduce_mean_MSE(mfccs_est, mfccs_label) / balance_coef # loss1/loss2 ~= 40
  return spec_loss, mfcc_loss

def balanced_MEL_AND_SPEC_MSE(y1,y2,spec_est,spec_label):
  '''
  spec_est:
    dim: [batch,time,frequence]
  spec_label:
    dim: [batch,time,frequence]
  '''
  spec_loss = reduce_sum_frame_batchsize_MSE(y1, y2)
  mel_est = tf_tool.melspec_form_realStft(spec_est, FLAGS.PARAM.FS, 20)
  mel_label = tf_tool.melspec_form_realStft(spec_label, FLAGS.PARAM.FS, 20)
  balance_coef = FLAGS.PARAM.MEL_BLANCE_COEF
  mel_loss = reduce_mean_MSE(mel_est, mel_label) / balance_coef # loss1/loss2 ~=3.2e8
  return spec_loss, mel_loss

def reduce_sum_frame_batchsize_MSE_Recurrent_Train(y_est1, y_est2, y_label):
  loss_stag1 = reduce_sum_frame_batchsize_MSE(y_est1,y_label)
  loss_stag2 = reduce_sum_frame_batchsize_MSE(y_est2,y_label)
  return 0.5*loss_stag1 + 0.5*loss_stag2

def reduce_mean_MSE_Recurrent_Train(y_est1, y_est2, y_label):
  loss_stag1 = reduce_mean_MSE(y_est1,y_label)
  loss_stag2 = reduce_mean_MSE(y_est2,y_label)
  return 0.5*loss_stag1 + 0.5*loss_stag2

def balanced_MFCC_AND_SPEC_MSE_Recurrent_Train(y_est1, y_est2, y_est_label,
                                               y_mag_est1, y_mag_est2, y_mag_label):
  loss_stag1 = balanced_MFCC_AND_SPEC_MSE(y_est1,y_est_label,y_mag_est1,y_mag_label)
  loss_stag2 = balanced_MFCC_AND_SPEC_MSE(y_est2,y_est_label,y_mag_est2,y_mag_label)
  return 0.5*loss_stag1 + 0.5*loss_stag2

def balanced_MEL_AND_SPEC_MSE_Recurrent_Train(y_est1, y_est2, y_est_label,
                                              y_mag_est1, y_mag_est2, y_mag_label):
  loss_stag1 = balanced_MEL_AND_SPEC_MSE(y_est1,y_est_label,y_mag_est1,y_mag_label)
  loss_stag2 = balanced_MEL_AND_SPEC_MSE(y_est2,y_est_label,y_mag_est2,y_mag_label)
  return 0.5*loss_stag1 + 0.5*loss_stag2

def reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE(y1, y2):
  loss1 = reduce_sum_frame_batchsize_MSE(y1, y2)
  low_frequence = 2000
  low_frequence_point = int(FLAGS.PARAM.OUTPUT_SIZE*(low_frequence/(FLAGS.PARAM.FS/2)))
  loss2 = reduce_sum_frame_batchsize_MSE(tf.slice(y1, [0, 0, 0], [-1, -1, low_frequence_point]),
                                         tf.slice(y2, [0, 0, 0], [-1, -1, low_frequence_point]))
  return loss1+loss2
