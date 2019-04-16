# speech enhancement
'''
training script:
    1_plural_train.py
'''
import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from utils import tf_tool
from losses import loss
from utils.tf_tool import norm_mag_spec, norm_logmag_spec, rm_norm_mag_spec, rm_norm_logmag_spec
from utils.tf_tool import normedLogmag2normedMag, normedMag2normedLogmag
from utils.tf_tool import lstm_cell, GRU_cell, CBHG, FrameProjection
import FLAGS
import numpy as np


class PluralMask_Model(object):
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
    assert(theta_x_batch is not None)
    if behavior != self.infer:
      assert(y_mag_spec_batch is not None)
      assert(theta_y_batch is not None)
    self._x_mag_spec = x_mag_spec_batch
    self._norm_x_mag_spec = norm_mag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)

    self._y_mag_spec = y_mag_spec_batch
    self._norm_y_mag_spec = norm_mag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)

    self._lengths = lengths_batch
    self._batch_size = tf.shape(self._lengths)[0]

    self._x_theta = theta_x_batch
    self._y_theta = theta_y_batch
    # self._norm_x_theta = self._x_theta/(2.0*FLAGS.PARAM.PI)+0.5
    # self._norm_y_theta = self._y_theta/(2.0*FLAGS.PARAM.PI)+0.5
    self._model_type = FLAGS.PARAM.MODEL_TYPE

    self.net_input = tf.concat([self._norm_x_mag_spec,self._x_theta],axis=-1)
    self._y_mag_labels = self._norm_y_mag_spec
    # self._y_theta_labels = self._norm_y_theta
    self._y_theta_labels = self._y_theta

    outputs = self.net_input
    if FLAGS.PARAM.INPUT_BN:
      with tf.variable_scope('Batch_Norm_Layer'):
        if_BRN = (FLAGS.PARAM.MVN_TYPE == 'BRN')
        if FLAGS.PARAM.SELF_BN:
          outputs = tf.layers.batch_normalization(outputs, training=True, renorm=if_BRN)
        else:
          outputs = tf.layers.batch_normalization(outputs,
                                                  training=(behavior == self.train or behavior == self.validation),
                                                  renorm=if_BRN)

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

    with tf.variable_scope("BiRNN"):
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
    if self._model_type.upper()[0] == 'B':  # bidirection
      rnn_output_num = FLAGS.PARAM.RNN_SIZE*2
      if FLAGS.PARAM.MODEL_TYPE == 'BLSTM' and (not (FLAGS.PARAM.LSTM_num_proj is None)):
        rnn_output_num = 2*FLAGS.PARAM.LSTM_num_proj
      in_size = rnn_output_num
    outputs = tf.reshape(outputs, [-1, in_size])
    out_size = FLAGS.PARAM.OUTPUT_SIZE
    with tf.variable_scope('fullconnectOut1'):
      out1_dense1 = tf.layers.Dense(out_size,activation='tanh')
      out1_dense2 = tf.layers.Dense(out_size//2,
                                    activation='relu' if FLAGS.PARAM.ReLU_MASK else None,
                                    bias_initializer=tf.constant_initializer(
                                        FLAGS.PARAM.INIT_MASK_VAL))
      self._mask1 = out1_dense2(out1_dense1(outputs))

    with tf.variable_scope('fullconnectOut2'):
      out2_dense1 = tf.layers.Dense(out_size,activation='tanh')
      out2_dense2 = tf.layers.Dense(out_size//2,
                                    activation='relu' if FLAGS.PARAM.ReLU_MASK else None,
                                    bias_initializer=tf.constant_initializer(
                                        FLAGS.PARAM.INIT_MASK_VAL))
      self._mask2 = out2_dense2(out2_dense1(outputs))

    self._mask1 = tf.reshape(self._mask1, [self._batch_size, -1, FLAGS.PARAM.OUTPUT_SIZE//2])
    self._mask2 = tf.reshape(self._mask2, [self._batch_size, -1, FLAGS.PARAM.OUTPUT_SIZE//2])

    self._mask = tf.concat([self._mask1,self._mask2],axis=-1)
    # endregion

    # mask type
    if FLAGS.PARAM.MASK_TYPE == 'PSM':
      self._y_mag_labels *= tf.cos(self._x_theta-self._y_theta)
    elif FLAGS.PARAM.MASK_TYPE == 'fixPSM':
      self._y_mag_labels *= (1.0+tf.cos(self._x_theta-self._y_theta))*0.5
    elif FLAGS.PARAM.MASK_TYPE == 'AcutePM':
      self._y_mag_labels *= tf.nn.relu(tf.cos(self._x_theta-self._y_theta))
    elif FLAGS.PARAM.MASK_TYPE == 'IRM':
      pass
    else:
      tf.logging.error('Mask type error.')
      exit(-1)

    # region get infer spec
    # self._y_est = self._mask*self.net_input # est->estimation
    # self._norm_y_mag_est = tf.slice(self._y_est,[0,0,0],[-1,-1,FLAGS.PARAM.FFT_DOT])
    # self._norm_y_theta_est = tf.slice(self._y_est,[0,0,FLAGS.PARAM.FFT_DOT],[-1,-1,-1])
    self._norm_y_mag_est = self._mask1*self._norm_x_mag_spec
    self._norm_y_theta_est = self._mask2*self._x_theta
    self._y_mag_est = rm_norm_mag_spec(self._norm_y_mag_est, FLAGS.PARAM.MAG_NORM_MAX)
    # self._y_theta_est = (self._norm_y_theta_est-0.5)*2.0*FLAGS.PARAM.PI
    self._y_theta_est = self._norm_y_theta_est
    # endregion

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if behavior == self.infer:
      return

    # region get LOSS
    if FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'SPEC_MSE': # log_mag and mag MSE
      self._mag_loss = loss.reduce_sum_frame_batchsize_MSE(self._norm_y_mag_est,self._y_mag_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "RELATED_MSE":
      self._mag_loss = loss.relative_reduce_sum_frame_batchsize_MSE(self._norm_y_mag_est,
                                                                    self._y_mag_labels,
                                                                    FLAGS.PARAM.RELATED_MSE_IGNORE_TH)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE":
      self._mag_loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE(self._norm_y_mag_est,
                                                                                self._y_mag_labels,
                                                                                FLAGS.PARAM.AUTO_RELATED_MSE_AXIS_FIT_DEG)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE_USE_COS":
      self._mag_loss = loss.cos_auto_ingore_relative_reduce_sum_frame_batchsize_MSE(self._norm_y_mag_est, self._y_mag_labels,
                                                                                    FLAGS.PARAM.COS_AUTO_RELATED_MSE_W)
    else:
      tf.logging.error('Magnitude_Loss type error.')
      exit(-1)

    if FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'COS':
      self._phase_loss = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(1.0-tf.cos(self._y_theta_est-self._y_theta_labels)),
                                                             FLAGS.PARAM.PHASE_LOSS_INDEX), 1))
    elif FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'MAG_WEIGHTED_COS':
      self._phase_loss = loss.magnitude_weighted_cos_deltaTheta(self._y_theta_est,
                                                                self._y_theta_labels,
                                                                self._norm_y_mag_spec,
                                                                index_=FLAGS.PARAM.PHASE_LOSS_INDEX)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'MIXMAG_WEIGHTED_COS':
      self._phase_loss = loss.magnitude_weighted_cos_deltaTheta(self._y_theta_est,
                                                                self._y_theta_labels,
                                                                self._norm_x_mag_spec,
                                                                index_=FLAGS.PARAM.PHASE_LOSS_INDEX)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'ABSOLUTE':
      self._phase_loss = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(self._y_theta_est-self._y_theta_labels),
                                                             FLAGS.PARAM.PHASE_LOSS_INDEX), 1))
    elif FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'MAG_WEIGHTED_ABSOLUTE':
      self._phase_loss = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(self._y_theta_est-self._y_theta_labels)*self._norm_y_mag_spec*10.0,
                                                             FLAGS.PARAM.PHASE_LOSS_INDEX), 1))
    elif FLAGS.PARAM.LOSS_FUNC_FOR_PHASE_SPEC == 'MIXMAG_WEIGHTED_ABSOLUTE':
      self._phase_loss = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(self._y_theta_est-self._y_theta_labels)*self._norm_x_mag_spec*10.0,
                                                             FLAGS.PARAM.PHASE_LOSS_INDEX), 1))
    else:
      tf.logging.error('Phase_Loss type error.')
      exit(-1)

    self._loss = self._mag_loss+self._phase_loss
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
  def y_mag_estimation(self):
    '''
    description: model outputs
    type: enhanced spectrum
    dims: [batch,time,frequence]
    '''
    return self._y_mag_est

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
    description: estimated angle of clean_spec
    type:
    dims: [batch, time, frequence]
    '''
    return self._y_theta_est


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
  def mag_loss(self):
    '''
    description: mag_loss
    type:
    dims: [0]
    '''
    return self._mag_loss


  @property
  def phase_loss(self):
    '''
    description: phase_loss
    type:
    dims: [0]
    '''
    return self._phase_loss


  @property
  def train_op(self):
    '''
    description: training operation node
    type: tensorflow computation node
    dims:
    '''
    return self._train_op
