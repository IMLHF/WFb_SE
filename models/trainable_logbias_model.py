# speech enhancement

import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from utils import tf_tool
from losses import loss
from utils.tf_tool import norm_mag_spec, norm_logmag_spec, rm_norm_mag_spec, rm_norm_logmag_spec
from utils.tf_tool import normedLogmag2normedMag, normedMag2normedLogmag
from utils.tf_tool import lstm_cell, GRU_cell
import FLAGS
import numpy as np


def auto_logbias(inputs, batch, input_finnal_dim):
  '''
  inputs: func inputs of calculate audio-suitable log_bias
    dims" [batch,time,features]
  '''
  in_size = input_finnal_dim
  with tf.variable_scope('fullconnectGetSuitableLogbias'):
    # attention layer
    with tf.variable_scope('attention_scorer'):
      weights_scorer = tf.get_variable('weights_scorer', [in_size, 1],
                                       initializer=tf.random_normal_initializer(stddev=0.01))
      biases_scorer = tf.get_variable('biases_scorer', [1],
                                      initializer=tf.constant_initializer(0.0))
      attention_alpha_vec = tf.matmul(tf.reshape(inputs, [-1, input_finnal_dim]),
                                      weights_scorer) + biases_scorer  # [batch*time,1]
      attention_alpha_vec = tf.reshape(attention_alpha_vec,[batch,1,-1]) # [batch,1,time]
      attention_alpha_vec = tf.nn.softmax(attention_alpha_vec, axis=-1)
      attened_vec = tf.reshape(tf.matmul(attention_alpha_vec, inputs), [batch,-1])# [batch,in_size]

    weights_logbias_fc = tf.get_variable('weights_logbias_fc', [in_size, 1],
                                         initializer=tf.random_normal_initializer(stddev=0.01))
    biases_logbias_fc = tf.get_variable('biases_logbias_fc', [1],
                                        initializer=tf.constant_initializer(0.0))
    logbias_net_out = tf.expand_dims(
      tf.matmul(attened_vec, weights_logbias_fc) + biases_logbias_fc, axis=-1)  # [batch,1,1]
    suitable_logbias = tf.nn.relu(logbias_net_out+FLAGS.PARAM.INIT_LOG_BIAS)

  return suitable_logbias # [batch,1,1]


class Trainable_Logbias_Model(object):
  infer = 'infer'
  train = 'train'
  validation = 'validation'

  def __init__(self,
               x_mag_spec_batch,
               lengths_batch,
               y_mag_spec_batch=None,
               theta_x_batch=None,
               theta_y_batch=None,
               behavior='train'):
    '''
    behavior = 'train/validation/infer'
    '''
    if behavior != self.infer:
      assert(y_mag_spec_batch is not None)
      assert(theta_x_batch is not None)
      assert(theta_y_batch is not None)
    self._x_mag_spec = x_mag_spec_batch
    self._norm_x_mag_spec = norm_mag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)

    self._y_mag_spec = y_mag_spec_batch
    self._norm_y_mag_spec = norm_mag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)

    self._lengths = lengths_batch
    self._batch_size = tf.shape(self._lengths)[0]

    self._x_theta = theta_x_batch
    self._y_theta = theta_y_batch
    self._model_type = FLAGS.PARAM.MODEL_TYPE

    if FLAGS.PARAM.INPUT_TYPE == 'mag':
      self.net_input = self._norm_x_mag_spec
    elif FLAGS.PARAM.INPUT_TYPE == 'logmag':
      tf.logging.error("Trainable_Logbias_Model: NNET input must be magnitude spectrum.")
      exit(-1)
    if FLAGS.PARAM.LABEL_TYPE == 'mag':
      tf.logging.error("Trainable_Logbias_Model: NNET labels must be log-magnitude spectrum.")
      exit(-1)

    outputs = self.net_input

    lstm_attn_cell = lstm_cell
    if behavior != self.infer and FLAGS.PARAM.KEEP_PROB < 1.0:
      def lstm_attn_cell(n_units, n_proj, act):
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(n_units, n_proj, act),
                                             output_keep_prob=FLAGS.PARAM.KEEP_PROB)

    GRU_attn_cell = GRU_cell
    if behavior != self.infer and FLAGS.PARAM.KEEP_PROB < 1.0:
      def GRU_attn_cell(n_units, act):
        return tf.contrib.rnn.DropoutWrapper(GRU_cell(n_units, act),
                                             output_keep_prob=FLAGS.PARAM.KEEP_PROB)

    if FLAGS.PARAM.MODEL_TYPE.upper() == 'BLSTM':
      with tf.variable_scope('BLSTM'):

        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE,
                            FLAGS.PARAM.LSTM_num_proj,
                            FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE,
                            FLAGS.PARAM.LSTM_num_proj,
                            FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)

        fw_cell = lstm_fw_cell._cells
        bw_cell = lstm_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=fw_cell,
            cells_bw=bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result

    if FLAGS.PARAM.MODEL_TYPE.upper() == 'BGRU':
      with tf.variable_scope('BGRU'):

        gru_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE,
                           FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)
        gru_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE,
                           FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)

        fw_cell = gru_fw_cell._cells
        bw_cell = gru_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=fw_cell,
            cells_bw=bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result

    # region full connection get mask
    # calcu rnn output size
    in_size = FLAGS.PARAM.RNN_SIZE
    mask = None
    if self._model_type.upper()[0] == 'B':  # bidirection
      rnn_output_num = FLAGS.PARAM.RNN_SIZE*2
      if FLAGS.PARAM.MODEL_TYPE == 'BLSTM' and (not (FLAGS.PARAM.LSTM_num_proj is None)):
        rnn_output_num = 2*FLAGS.PARAM.LSTM_num_proj
      in_size = rnn_output_num
    outputs = tf.reshape(outputs, [-1, in_size])
    out_size = FLAGS.PARAM.OUTPUT_SIZE
    with tf.variable_scope('fullconnectOut'):
      weights = tf.get_variable('weights1', [in_size, out_size],
                                initializer=tf.random_normal_initializer(stddev=0.01))
      biases = tf.get_variable('biases1', [out_size],
                               initializer=tf.constant_initializer(0.0))
    if FLAGS.PARAM.TIME_NOSOFTMAX_ATTENTION:
      with tf.variable_scope('fullconnectCoef'):
        weights_coef = tf.get_variable('weights_coef', [in_size, 1],
                                       initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01))
        biases_coef = tf.get_variable('biases_coef', [1],
                                      initializer=tf.constant_initializer(0.0))
      mask = tf.multiply(tf.nn.softmax(tf.reshape(tf.matmul(outputs, weights) + biases, [self._batch_size,-1])),
                         tf.nn.relu(tf.reduce_sum(tf.matmul(outputs, weights_coef) + biases_coef, axis=[-1,-2])))
    else:
      mask = tf.nn.relu(tf.matmul(outputs, weights) + biases)
    self._mask = tf.reshape(
        mask, [self._batch_size, -1, FLAGS.PARAM.OUTPUT_SIZE])
    # endregion

    # region get suitable log_bias
    outputs = tf.reshape(outputs, [self._batch_size, -1, in_size])
    self._log_bias = auto_logbias(outputs,self._batch_size,in_size)
    self._real_logbias = tf.add(self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    # endregion
    self._norm_x_logmag_spec = norm_logmag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    self._norm_y_logmag_spec = norm_logmag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)

    # region prepare y_estimation and y_labels
    self._y_labels = self._norm_y_logmag_spec
    if FLAGS.PARAM.TRAINING_MASK_POSITION == 'mag':
      self._y_estimation = self._mask*self._norm_x_mag_spec
    elif FLAGS.PARAM.TRAINING_MASK_POSITION == 'logmag':
      self._y_estimation = self._mask*self._norm_x_logmag_spec
    if FLAGS.PARAM.MASK_TYPE == 'PSM':
      self._y_labels *= tf.cos(self._x_theta-self._y_theta)
    elif FLAGS.PARAM.MASK_TYPE == 'IRM':
      pass
    else:
      tf.logging.error('Mask type error.')
      exit(-1)

    # region get infer spec
    if FLAGS.PARAM.DECODING_MASK_POSITION != FLAGS.PARAM.TRAINING_MASK_POSITION:
      print('Error, DECODING_MASK_POSITION should be equal to TRAINING_MASK_POSITION when use trainable_logbias_model.')
    if FLAGS.PARAM.DECODING_MASK_POSITION == 'mag':
      self._y_mag_estimation = rm_norm_mag_spec(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX)
    elif FLAGS.PARAM.DECODING_MASK_POSITION == 'logmag':
      self._y_mag_estimation = rm_norm_logmag_spec(self._y_estimation,
                                                   FLAGS.PARAM.MAG_NORM_MAX,
                                                   self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    '''
    _y_mag_estimation is estimated mag_spec
    _y_estimation is loss_targe,  logmag_spec
    '''
    # endregion

    if FLAGS.PARAM.TRAINING_MASK_POSITION != FLAGS.PARAM.LABEL_TYPE:
      if FLAGS.PARAM.LABEL_TYPE == 'mag':
        self._y_estimation = normedLogmag2normedMag(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                    self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
      elif FLAGS.PARAM.LABEL_TYPE == 'logmag':
        self._y_estimation = normedMag2normedLogmag(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                    self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    # endregion

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)

    if behavior == self.infer:
      return

    # region get LOSS
    if FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'SPEC_MSE': # log_mag and mag MSE
      self._loss = loss.reduce_sum_frame_batchsize_MSE(self._y_estimation,self._y_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'MFCC_SPEC_MSE':
      self._loss1, self._loss2 = loss.balanced_MFCC_AND_SPEC_MSE(self._y_estimation, self._y_labels,
                                                                 self._y_mag_estimation, self._y_mag_spec)
      self._loss = FLAGS.PARAM.SPEC_LOSS_COEF*self._loss1 + FLAGS.PARAM.MFCC_LOSS_COEF*self._loss2
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'MEL_MAG_MSE':
      self._loss1, self._loss2 = loss.balanced_MEL_AND_SPEC_MSE(self._y_estimation, self._y_labels,
                                                                self._y_mag_estimation, self._y_mag_spec)
      self._loss = FLAGS.PARAM.SPEC_LOSS_COEF*self._loss1 + FLAGS.PARAM.MEL_LOSS_COEF*self._loss2
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "SPEC_MSE_LOWF_EN":
      self._loss = loss.reduce_sum_frame_batchsize_MSE(self._y_estimation, self._y_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "FAIR_SPEC_MSE":
      self._loss = loss.fair_reduce_sum_frame_batchsize_MSE(self._y_estimation, self._y_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "SPEC_MSE_FLEXIBLE_POW_C":
      self._loss = loss.reduce_sum_frame_batchsize_MSE_EmphasizeLowerValue(self._y_estimation,
                                                                           self._y_labels,
                                                                           FLAGS.PARAM.POW_COEF)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "LOG_BIAS_DE":
      self._loss = loss.logbias_de_reduce_sum_frame_batchsize_MSE(self._y_estimation,
                                                                  self._y_labels,
                                                                  self._real_logbias)
    else:
      print('Loss type error.')
      exit(-1)
    # endregion

    if behavior == self.validation:
      '''
      val model cannot train.
      '''
      return
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      FLAGS.PARAM.CLIP_NORM)
    optimizer = tf.train.AdamOptimizer(self.lr)
    #optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name='new_learning_rate')
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def log_bias(self):
    '''
    description: logbias of log(x+logbias)
    type: >0
    dims: [batch,1,1]
    '''
    return self._real_logbias


  @property
  def y_mag_estimation(self):
    '''
    description: model outputs
    type: enhanced spectrum
    dims: [batch,time,frequence]
    '''
    return self._y_mag_estimation

  @property
  def x_mag(self):
    '''
    description: model inputs
    type: mixture spectrum
    dims: [batch,time,frequence]
    '''
    return self._x_mag_spec

  @property
  def y_mag(self):
    '''
    description: trainning reference
    type: clean spectrum
    dims: [batch,time,frequence]
    '''
    return self._y_mag_spec

  @property
  def mask(self):
    '''
    description: wiener filtering mat
    type:
    dims: same to spectrum
    '''
    return self._mask

  @property
  def x_theta(self):
    '''
    description: angle of inputs
    type:
    dims: [batch, time, frequence]
    '''
    return self._x_theta

  @property
  def y_theta(self):
    '''
    description: angle of labels
    type:
    dims: [batch, time, frequence]
    '''
    return self._y_theta

  @property
  def y_theta_estimation(self):
    '''
    estimate y_theta_est placeholder
    '''
    return self._lengths

  @property
  def lengths(self):
    '''
    description: dynamic time length
    type: an int list
    dims: [batch]
    '''
    return self._lengths

  @property
  def batch_size(self):
    '''
    description: dynamic batch_size
    type: an int number
    dims: [0]
    '''
    return self._batch_size

  @property
  def lr(self):
    '''
    description: learning rate
    type:
    dims: [0]
    '''
    return self._lr

  @property
  def loss(self):
    '''
    description: model loss
    type:
    dims: [0]
    '''
    return self._loss

  @property
  def train_op(self):
    '''
    description: training operation node
    type: tensorflow computation node
    dims:
    '''
    return self._train_op
