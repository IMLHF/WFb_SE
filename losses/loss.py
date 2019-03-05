import tensorflow as tf
import numpy as np
import FLAGS

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
