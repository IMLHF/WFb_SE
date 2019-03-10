import tensorflow as tf
import numpy as np
import FLAGS
from utils import tf_tool

def reduce_sum_frame_batchsize_MSE(y1, y2):
  cost = tf.reduce_mean(tf.reduce_sum(tf.pow(y1-y2, 2), 1), 1)
  return tf.reduce_sum(cost)


def reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE(y1, y2):
  loss1 = reduce_sum_frame_batchsize_MSE(y1, y2)
  low_frame = 2000
  low_frame_point = int(FLAGS.PARAM.OUTPUT_SIZE*(low_frame/(FLAGS.PARAM.FS/2)))
  loss2 = reduce_sum_frame_batchsize_MSE(tf.slice(y1, [0, 0, 0], [-1, -1, low_frame_point]),
                                         tf.slice(y2, [0, 0, 0], [-1, -1, low_frame_point]))
  return loss1+loss2

def reduce_sum_frame_batchsize_MFCC_AND_SPEC_MSE(y1,y2,spec_est,spec_label):
  '''
  spec_est:
    dim: [batch,time,frequence]
  spec_label:
    dim: [batch,time,frequence]
  '''
  mfccs_est = tf_tool.mfccs_form_realStft(spec_est, FLAGS.PARAM.FS, 20, 13)
  mfccs_label = tf_tool.mfccs_form_realStft(spec_label, FLAGS.PARAM.FS, 20, 13)
  loss1 = reduce_sum_frame_batchsize_MSE(mfccs_est, mfccs_label)
  loss2 = reduce_sum_frame_batchsize_MSE(y1, y2)
  return loss1,loss2
