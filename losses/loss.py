import tensorflow as tf
import numpy as np
import FLAGS
from utils import tf_tool


def MEL_AUTO_RELATIVE_MSE(spec_est,spec_label,mel_num,AFD):
  '''
  spec_est:
    dim: [batch,time,frequence]
  spec_label:
    dim: [batch,time,frequence]
  '''
  mel_est = tf_tool.melspec_form_realStft(spec_est, FLAGS.PARAM.FS, mel_num)
  mel_label = tf_tool.melspec_form_realStft(spec_label, FLAGS.PARAM.FS, mel_num)
  mel_loss = auto_ingore_relative_reduce_sum_frame_batchsize_MSE(mel_est, mel_label, AFD)
  return mel_loss


def magnitude_weighted_cos_deltaTheta(theta1,theta2,mag_spec,index_=2.0):
  cost = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(tf.multiply(1.0-tf.cos(theta1-theta2), mag_spec*10.0)),
                                             index_),
                                      1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v8(y_est,y_true,A,B,C1,C2):
  '''
    [|y-y_|/(A*(|y|+|y_|)^C1+B)]^C2
  '''
  refer_true = A*tf.pow((tf.abs(y_true)+tf.abs(y_est)),C1)+B
  relative_loss = tf.abs(y_est-y_true)/refer_true
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, C2), 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v7(y_est,y_true,A1,A2,B,C1,C2):
  refer = tf.pow(A1*tf.sqrt(tf.abs(y_true))*tf.sqrt(tf.abs(y_true)+tf.abs(y_est))+A2*(tf.abs(y_est)+tf.abs(y_true)),C1)+B
  relative_loss = tf.abs(y_est-y_true)/refer
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, C2), 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v6(y1,y2,A,B,C1,C2):
  refer_mul = tf.pow(tf.sqrt(tf.abs(y1))*tf.sqrt(tf.abs(y2)),C1)*A+B
  relative_loss = tf.abs(y1-y2)/refer_mul
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, C2), 1), 1))
  return cost


# AUTO_RELATED_MSE5
def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v5(y_est,y_true,index_=2.0):
  reluYsubY_ = tf.nn.relu(y_true-y_est)
  part1 = tf.pow(reluYsubY_ / (2.0*tf.abs(y_true)+reluYsubY_), index_)
  part2 = tf.pow(reluYsubY_ / (tf.abs(y_true)+tf.nn.relu(tf.abs(y_true)-y_true+y_est)), index_)
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(part1+part2, 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v4(y_est,y_true,axis_fit_degree,index_=2.0):
  refer = 1.0/axis_fit_degree+(1.0-1.0/axis_fit_degree)*(tf.abs(y_true)+tf.nn.relu(tf.sign(y_true)*y_est))
  relative_loss = tf.abs(y_est-y_true)/refer
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v3(y_est,y_true,A,B,C1,C2):
  '''
    [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  refer_true = tf.pow(A*tf.abs(y_true),C1)+B
  relative_loss = tf.abs(y_est-y_true)/refer_true
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, C2), 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v2(y1,y2,axis_fit_degree,linear_broker,index_=2.0):
  # refer_sum = tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12)
  # small_val_debuff = tf.pow(refer_sum*axis_fit_degree*1.0,-1.0)+1.0-tf.pow(axis_fit_degree*1.0,-1.0)
  # relative_loss = tf.pow(tf.abs(y1-y2),linear_broker)/refer_sum/small_val_debuff
  refer = 1.0/axis_fit_degree+(1.0-1.0/axis_fit_degree)*(tf.abs(y1)+tf.abs(y2))
  relative_loss = tf.pow(tf.abs(y1-y2),linear_broker)/refer
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 1), 1))
  return cost


def auto_ingore_relative_reduce_sum_frame_batchsize_MSE(y1,y2,axis_fit_degree,index_=2.0):
  # refer_sum = tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12)
  # small_val_debuff = tf.pow(refer_sum*axis_fit_degree*1.0,-1.0)+1.0-tf.pow(axis_fit_degree*1.0,-1.0)
  # relative_loss = tf.abs(y1-y2)/refer_sum/small_val_debuff
  refer = 1.0/axis_fit_degree+(1.0-1.0/axis_fit_degree)*(tf.abs(y1)+tf.abs(y2))
  relative_loss = tf.abs(y1-y2)/refer
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 1), 1))
  return cost


def relative_reduce_sum_frame_batchsize_MSE(y1,y2,ignore_threshold):
  # cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2)/tf.maximum(tf.abs(y1)+tf.abs(y2),ignore_threshold), 2), 1), 1))
  refer = tf.abs(y1)+tf.abs(y2)+ignore_threshold
  relative_loss = tf.abs(y1-y2)/refer
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, 2.0), 1), 1))
  return cost


def cos_auto_ingore_relative_reduce_sum_frame_batchsize_MSE(y1,y2,w,index_=2.0):
  refer_sum = tf.abs(y1)+tf.abs(y2)+1e-12
  relative_loss = tf.abs(y1-y2)/refer_sum
  small_val_debuff_useCOS = (1.0-tf.cos(refer_sum*w))*0.5
  relative_cost = relative_loss*small_val_debuff_useCOS
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(relative_cost, index_), 1), 1))
  return cost


def logbias_de_reduce_sum_frame_batchsize_MSE(y1,y2,logbias):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2), 2), -1), -1)+tf.pow(logbias/5000,2))
  return cost


def reduce_sum_frame_batchsize_MSE_EmphasizeLowerValue(y1, y2, pow_coef):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2), pow_coef), 1), 1))
  return cost

def reduce_sum_frame_batchsize_MSE(y1, y2):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2), 2), 1), 1))
  return cost

def fair_reduce_sum_frame_batchsize_MSE(y1, y2):
  cost = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(y1-y2)+1, 2)-1, 1), 1))
  return cost

def reduce_mean_MSE(y1, y2):
  cost = tf.reduce_mean(tf.pow(tf.abs(y1-y2), 2))
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
  mel_est = tf_tool.melspec_form_realStft(spec_est, FLAGS.PARAM.FS, 80)
  mel_label = tf_tool.melspec_form_realStft(spec_label, FLAGS.PARAM.FS, 80)
  balance_coef = FLAGS.PARAM.MEL_BLANCE_COEF
  mel_loss = reduce_mean_MSE(mel_est, mel_label) / balance_coef # loss1/loss2 ~=3.2e7
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
  low_frequence_point = int(FLAGS.PARAM.FFT_DOT*(low_frequence/(FLAGS.PARAM.FS/2)))
  loss2 = reduce_sum_frame_batchsize_MSE(tf.slice(y1, [0, 0, 0], [-1, -1, low_frequence_point]),
                                         tf.slice(y2, [0, 0, 0], [-1, -1, low_frequence_point]))
  return loss1+loss2
