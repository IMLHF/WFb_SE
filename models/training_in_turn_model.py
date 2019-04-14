# speech enhancement

import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from utils import tf_tool
from losses import loss
from utils.tf_tool import norm_mag_spec, norm_logmag_spec, rm_norm_mag_spec, rm_norm_logmag_spec
from utils.tf_tool import normedLogmag2normedMag, normedMag2normedLogmag
from utils.tf_tool import lstm_cell, GRU_cell, sum_attention
import FLAGS
import numpy as np


class ALTER_Training_Model(object):
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
      self.logbias_net_input = self._norm_x_mag_spec
    elif FLAGS.PARAM.INPUT_TYPE == 'logmag':
      tf.logging.error("Training_In_Turn_Model: NNET input must be magnitude spectrum.")
      exit(-1)

    # region training dropout
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
    # endregion

    # region logbias net
    with tf.variable_scope('logbias_net'):
      logbias_net_outputs = self.logbias_net_input
      if FLAGS.PARAM.MODEL_TYPE.upper() == 'BLSTM':
        with tf.variable_scope('BLSTM_logbias'):

          lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
              [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE_LOGBIAS,
                              FLAGS.PARAM.LSTM_num_proj_LOGBIAS,
                              FLAGS.PARAM.LSTM_ACTIVATION_LOGBIAS)
               for _ in range(FLAGS.PARAM.RNN_LAYER_LOGBIAS)], state_is_tuple=True)
          lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
              [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE_LOGBIAS,
                              FLAGS.PARAM.LSTM_num_proj_LOGBIAS,
                              FLAGS.PARAM.LSTM_ACTIVATION_LOGBIAS)
               for _ in range(FLAGS.PARAM.RNN_LAYER_LOGBIAS)], state_is_tuple=True)

          fw_cell_logbiasnet = lstm_fw_cell._cells
          bw_cell_logbiasnet = lstm_bw_cell._cells

      if FLAGS.PARAM.MODEL_TYPE.upper() == 'BGRU':
        with tf.variable_scope('BGRU_logbias'):

          gru_fw_cell = tf.contrib.rnn.MultiRNNCell(
              [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE_LOGBIAS,
                             FLAGS.PARAM.LSTM_ACTIVATION_LOGBIAS)
               for _ in range(FLAGS.PARAM.RNN_LAYER_LOGBIAS)], state_is_tuple=True)
          gru_bw_cell = tf.contrib.rnn.MultiRNNCell(
              [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE_LOGBIAS,
                             FLAGS.PARAM.LSTM_ACTIVATION_LOGBIAS)
               for _ in range(FLAGS.PARAM.RNN_LAYER_LOGBIAS)], state_is_tuple=True)

          fw_cell_logbiasnet = gru_fw_cell._cells
          bw_cell_logbiasnet = gru_bw_cell._cells

      # dynamic rnn
      result = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw=fw_cell_logbiasnet,
          cells_bw=bw_cell_logbiasnet,
          inputs=logbias_net_outputs,
          dtype=tf.float32,
          sequence_length=self._lengths)
      logbias_net_outputs, fw_final_states, bw_final_states = result

      logbias_biRnn_out_size = FLAGS.PARAM.RNN_SIZE_LOGBIAS*2
      # attend_fea = sum_attention_v2(logbias_net_outputs,self._batch_size,logbias_biRnn_out_size)
      # print(np.shape(fw_final_states),np.shape(bw_final_states),np.shape(logbias_net_outputs))
      # attend_fea = sum_attention_with_final_state(logbias_net_outputs,
      #                                             tf.concat(-1, [fw_final_states,
      #                                                            bw_final_states]),
      #                                             logbias_biRnn_out_size, 1024)
      attend_fea = sum_attention(logbias_net_outputs,
                                 logbias_biRnn_out_size, 1024)

      with tf.variable_scope('fullconnectSuitableLogbias'):
        weights_logbias_fc = tf.get_variable('weights_logbias_fc', [logbias_biRnn_out_size, 1],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
        biases_logbias_fc = tf.get_variable('biases_logbias_fc', [1],
                                            initializer=tf.constant_initializer(0.0))
        logbias_net_out = tf.expand_dims(
          tf.matmul(attend_fea, weights_logbias_fc) + biases_logbias_fc, axis=-1)  # [batch,1,1]
        self._log_bias = tf.nn.relu(logbias_net_out+FLAGS.PARAM.INIT_LOG_BIAS)

      self._real_logbias = tf.add(self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    # endregion

    self._norm_x_logmag_spec = norm_logmag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    self._norm_y_logmag_spec = norm_logmag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)

    # region mask net
    with tf.variable_scope('mask_net'):
      mask_net_outputs = self._norm_x_logmag_spec
      if FLAGS.PARAM.MODEL_TYPE.upper() == 'BLSTM':
        with tf.variable_scope('BLSTM_mask'):

          lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
              [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE_MASK,
                              FLAGS.PARAM.LSTM_num_proj_MASK,
                              FLAGS.PARAM.LSTM_ACTIVATION_MASK)
               for _ in range(FLAGS.PARAM.RNN_LAYER_MASK)], state_is_tuple=True)
          lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
              [lstm_attn_cell(FLAGS.PARAM.RNN_SIZE_MASK,
                              FLAGS.PARAM.LSTM_num_proj_MASK,
                              FLAGS.PARAM.LSTM_ACTIVATION_MASK)
               for _ in range(FLAGS.PARAM.RNN_LAYER_MASK)], state_is_tuple=True)

          fw_cell_masknet = lstm_fw_cell._cells
          bw_cell_masknet = lstm_bw_cell._cells

      if FLAGS.PARAM.MODEL_TYPE.upper() == 'BGRU':
        with tf.variable_scope('BGRU_mask'):

          gru_fw_cell = tf.contrib.rnn.MultiRNNCell(
              [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE,
                             FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)
          gru_bw_cell = tf.contrib.rnn.MultiRNNCell(
              [GRU_attn_cell(FLAGS.PARAM.RNN_SIZE,
                             FLAGS.PARAM.LSTM_ACTIVATION) for _ in range(FLAGS.PARAM.RNN_LAYER)], state_is_tuple=True)

          fw_cell_masknet = gru_fw_cell._cells
          bw_cell_masknet = gru_bw_cell._cells

      # dynamic rnn
      result = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw=fw_cell_masknet,
          cells_bw=bw_cell_masknet,
          inputs=mask_net_outputs,
          dtype=tf.float32,
          sequence_length=self._lengths)

      mask_net_outputs, fw_final_states, bw_final_states = result
      mask_biRnn_output_size = FLAGS.PARAM.RNN_SIZE_MASK*2
      flatten_outputs = tf.reshape(mask_net_outputs, [-1, mask_biRnn_output_size])
      out_size = FLAGS.PARAM.OUTPUT_SIZE
      with tf.variable_scope('fullconnectMask'):
        weights = tf.get_variable('weights1', [mask_biRnn_output_size, out_size],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases1', [out_size],
                                 initializer=tf.constant_initializer(0.0))
      mask = tf.nn.relu(tf.matmul(flatten_outputs, weights) + biases)
      self._mask = tf.reshape(
          mask, [self._batch_size, -1, FLAGS.PARAM.OUTPUT_SIZE])
    # endregion

    # region prepare y_estimation and y_labels
    self._y_mag_labels = self._norm_y_mag_spec
    self._y_logmag_labels = self._norm_y_logmag_spec
    if FLAGS.PARAM.TRAINING_MASK_POSITION == 'mag':
      self._y_normed_mag_estimation = self._mask*self._norm_x_mag_spec
      self._y_normed_logmag_estimation = normedMag2normedLogmag(self._y_normed_mag_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                                self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    elif FLAGS.PARAM.TRAINING_MASK_POSITION == 'logmag':
      self._y_normed_logmag_estimation = self._mask*self._norm_x_logmag_spec
      self._y_normed_mag_estimation = normedLogmag2normedMag(self._y_normed_logmag_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                             self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    if FLAGS.PARAM.MASK_TYPE == 'PSM':
      self._y_mag_labels *= tf.cos(self._x_theta-self._y_theta)
      self._y_logmag_labels *= tf.cos(self._x_theta-self._y_theta)
    elif FLAGS.PARAM.MASK_TYPE == 'IRM':
      pass
    else:
      tf.logging.error('Mask type error.')
      exit(-1)

    # region get infer spec
    if FLAGS.PARAM.DECODING_MASK_POSITION != FLAGS.PARAM.TRAINING_MASK_POSITION:
      print('Error, DECODING_MASK_POSITION should be equal to TRAINING_MASK_POSITION when use training_in_turn_model.')
    if FLAGS.PARAM.DECODING_MASK_POSITION == 'mag':
      self._y_mag_estimation = rm_norm_mag_spec(self._y_normed_mag_estimation, FLAGS.PARAM.MAG_NORM_MAX)
    elif FLAGS.PARAM.DECODING_MASK_POSITION == 'logmag':
      self._y_mag_estimation = rm_norm_logmag_spec(self._y_normed_logmag_estimation,
                                                   FLAGS.PARAM.MAG_NORM_MAX,
                                                   self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    '''
    _y_mag_estimation is estimated mag_spec
    '''
    # endregion

    # endregion

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)

    if behavior == self.infer:
      return

    # region get LOSS
    if FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'SPEC_MSE':  # log_mag and mag MSE
      self._logbiasnet_loss = loss.relative_reduce_sum_frame_batchsize_MSE(
          self._y_normed_mag_estimation, self._y_mag_labels, 1e-6)
      self._masknet_loss = loss.reduce_sum_frame_batchsize_MSE(
        self._y_normed_logmag_estimation,self._y_logmag_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "SPEC_MSE_FLEXIBLE_POW_C":
      self._logbiasnet_loss = loss.reduce_sum_frame_batchsize_MSE_EmphasizeLowerValue(self._y_normed_mag_estimation,
                                                                                      self._y_mag_labels,
                                                                                      FLAGS.PARAM.POW_COEF)
      self._masknet_loss = loss.reduce_sum_frame_batchsize_MSE_EmphasizeLowerValue(self._y_normed_logmag_estimation,
                                                                                   self._y_logmag_labels,
                                                                                   FLAGS.PARAM.POW_COEF)
    else:
      print('Loss type error.')
      exit(-1)
    # endregion

    if behavior == self.validation:
      '''
      val model cannot train.
      '''
      return
    self._lr_logbiasnet = tf.Variable(0.0, trainable=False)
    self._lr_masknet = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    logbias_vars = [var for var in tvars if 'logbias_net' in var.name]
    mask_vars = [var for var in tvars if 'mask_net' in var.name]
    logbiasnet_grads, _ = tf.clip_by_global_norm(tf.gradients(self._logbiasnet_loss, logbias_vars),
                                                 FLAGS.PARAM.CLIP_NORM)
    masknet_grads, _ = tf.clip_by_global_norm(tf.gradients(self._masknet_loss, mask_vars),
                                              FLAGS.PARAM.CLIP_NORM)
    optimizer_logbiasnet = tf.train.AdamOptimizer(self.lr_logbiasnet)
    optimizer_masknet = tf.train.AdamOptimizer(self.lr_masknet)
    #optimizer = tf.train.GradientDescentOptimizer(self.lr)
    # all_grads = [grad for grad in logbiasnet_grads]
    # for grad in masknet_grads:
    #   all_grads.append(grad)
    # all_vars = [var for var in logbias_vars]
    # for var in mask_vars:
    #   all_vars.append(var)
    train_logbiasnet = optimizer_logbiasnet.apply_gradients(zip(logbiasnet_grads, logbias_vars))
    train_masknet = optimizer_masknet.apply_gradients(zip(masknet_grads, mask_vars))
    if FLAGS.PARAM.TRAIN_TYPE == 'BOTH':
      self._train_op = [train_logbiasnet, train_masknet]
    elif FLAGS.PARAM.TRAIN_TYPE == 'LOGBIASNET':
      self._train_op = train_logbiasnet
    elif FLAGS.PARAM.TRAIN_TYPE == 'MASKNET':
      self._train_op = train_masknet

    self._new_lr_logbiasnet = tf.placeholder(
        tf.float32, shape=[], name='new_learning_rate1')
    self._new_lr_masknet = tf.placeholder(
        tf.float32, shape=[], name='new_learning_rate2')
    self._lr_update = [tf.assign(self._lr_logbiasnet, self._new_lr_logbiasnet),
                       tf.assign(self._lr_masknet, self._new_lr_masknet)]

  def assign_lr(self, session, lr_logbiasnet, lr_masknet):
    session.run(self._lr_update, feed_dict={self._new_lr_logbiasnet: lr_logbiasnet,
                                            self._new_lr_masknet: lr_masknet})

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
  def lr_logbiasnet(self):
    '''
    description: learning rate
    type:
    dims: [0]
    '''
    return self._lr_logbiasnet


  @property
  def lr_masknet(self):
    '''
    description: learning rate
    type:
    dims: [0]
    '''
    return self._lr_masknet


  @property
  def logbiasnet_loss(self):
    '''
    description: model loss1
    type:
    dims: [0]
    '''
    return self._logbiasnet_loss


  @property
  def masknet_loss(self):
    '''
    description: model loss2
    type:
    dims: [0]
    '''
    return self._masknet_loss


  @property
  def train_op(self):
    '''
    description: training operation node
    type: tensorflow computation node
    dims:
    '''
    return self._train_op
