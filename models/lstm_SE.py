# speech enhancement

import tensorflow as tf
import time
import sys
from tensorflow.contrib.rnn.python.ops import rnn
from utils import tf_tool
from FLAGS import PARAM

DEFAULT_LOG_BIAS = 1e-12

# trunc mag and dispersion to [0,1]
def norm_mag_spec(mag_spec):
  mag_spec = tf.clip_by_value(mag_spec, 0, PARAM.MAG_NORM_MAX)
  normed_mag = mag_spec / (PARAM.MAG_NORM_MAX - 0)
  return normed_mag

# add bias and logarithm to mag, dispersion to [0,1]
def norm_logmag_spec(mag_spec, log_bias):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS+PARAM.MAG_NORM_MAX) / tf.log(10.0)

  mag_spec = tf.clip_by_value(mag_spec, 0, PARAM.MAG_NORM_MAX)
  logmag_spec = tf.log(mag_spec+tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS)/tf.log(10.0)
  logmag_spec -= LOG_NORM_MIN
  normed_logmag = logmag_spec / (LOG_NORM_MAX - LOG_NORM_MIN)
  return normed_logmag

# Inverse process of norm_mag_spec()
def rm_norm_mag_spec(normed_mag):
  normed_mag *= (PARAM.MAG_NORM_MAX - 0)
  mag_spec = normed_mag
  return mag_spec

# Inverse process of norm_logmag_spec()
def rm_norm_logmag_spec(normed_logmag, log_bias):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS+PARAM.MAG_NORM_MAX) / tf.log(10.0)

  normed_logmag *= (LOG_NORM_MAX - LOG_NORM_MIN)
  normed_logmag += LOG_NORM_MIN
  normed_logmag *= tf.log(10.0)
  mag_spec = tf.exp(normed_logmag) - DEFAULT_LOG_BIAS - tf.nn.relu(log_bias)
  return mag_spec

#
def normedMag2normedLogmag(normed_mag, log_bias):
  return norm_logmag_spec(rm_norm_mag_spec(normed_mag), log_bias)

#
def normedLogmag2normedMag(normed_logmag, log_bias):
  return norm_mag_spec(rm_norm_logmag_spec(normed_logmag, log_bias))


class SE_MODEL(object):
  def __init__(self,
               x_mag_spec_batch,
               lengths_batch,
               y_mag_spec_batch=None,
               theta_x_batch=None,
               theta_y_batch=None,
               infer=False):
    self._log_bias = tf.get_variable('logbias', [1], trainable=PARAM.LOG_BIAS_TRAINABEL,
                                     initializer=tf.constant_initializer(PARAM.INIT_LOG_BIAS))
    self._real_logbias = self._log_bias + DEFAULT_LOG_BIAS
    self._inputs = x_mag_spec_batch
    self._x_mag_spec = self.inputs
    self._norm_x_mag_spec = norm_mag_spec(self._x_mag_spec)
    self._norm_x_logmag_spec = norm_logmag_spec(self._x_mag_spec, self._log_bias)

    if not infer:
      self._y_mag_spec = y_mag_spec_batch
      self._norm_y_mag_spec = norm_mag_spec(self._y_mag_spec)
      self._norm_y_logmag_spec = norm_logmag_spec(self._y_mag_spec, self._log_bias)

    self._lengths = lengths_batch

    self.batch_size = tf.shape(self._lengths)[0]
    self._model_type = PARAM.MODEL_TYPE

    if PARAM.INPUT_TYPE == 'mag':
      self.net_input = self._norm_x_mag_spec
    elif PARAM.INPUT_TYPE == 'logmag':
      self.net_input = self._norm_x_logmag_spec

    if not infer:
      if PARAM.LABEL_TYPE == 'mag':
        self._labels = self._norm_y_mag_spec
      elif PARAM.LABEL_TYPE == 'logmag':
        self._labels = self._norm_y_logmag_spec

    outputs = self.net_input

    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(
          PARAM.RNN_SIZE, forget_bias=1.0, use_peepholes=True,
          num_proj=PARAM.LSTM_num_proj,
          initializer=tf.contrib.layers.xavier_initializer(),
          state_is_tuple=True, activation=PARAM.LSTM_ACTIVATION)
    lstm_attn_cell = lstm_cell
    if not infer and PARAM.KEEP_PROB < 1.0:
      def lstm_attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=PARAM.KEEP_PROB)

    def GRU_cell():
      return tf.contrib.rnn.GRUCell(
          PARAM.RNN_SIZE,
          # kernel_initializer=tf.contrib.layers.xavier_initializer(),
          activation=PARAM.LSTM_ACTIVATION)
    GRU_attn_cell = lstm_cell
    if not infer and PARAM.KEEP_PROB < 1.0:
      def GRU_attn_cell():
        return tf.contrib.rnn.DropoutWrapper(GRU_cell(), output_keep_prob=PARAM.KEEP_PROB)

    if PARAM.MODEL_TYPE.upper() == 'BLSTM':
      with tf.variable_scope('BLSTM'):

        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell() for _ in range(PARAM.RNN_LAYER)], state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_attn_cell() for _ in range(PARAM.RNN_LAYER)], state_is_tuple=True)

        lstm_fw_cell = lstm_fw_cell._cells
        lstm_bw_cell = lstm_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=lstm_fw_cell,
            cells_bw=lstm_bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result
    if PARAM.MODEL_TYPE.upper() == 'BGRU':
      with tf.variable_scope('BGRU'):

        gru_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell() for _ in range(PARAM.RNN_LAYER)], state_is_tuple=True)
        gru_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [GRU_attn_cell() for _ in range(PARAM.RNN_LAYER)], state_is_tuple=True)

        gru_fw_cell = gru_fw_cell._cells
        gru_bw_cell = gru_bw_cell._cells
        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=gru_fw_cell,
            cells_bw=gru_bw_cell,
            inputs=outputs,
            dtype=tf.float32,
            sequence_length=self._lengths)
        outputs, fw_final_states, bw_final_states = result

    with tf.variable_scope('fullconnectOut'):
      in_size = PARAM.RNN_SIZE
      if self._model_type.upper()[0] == 'B':  # bidirection
        rnn_output_num = PARAM.RNN_SIZE*2
        if PARAM.MODEL_TYPE == 'BLSTM' and (not (PARAM.LSTM_num_proj is None)):
          rnn_output_num = 2*PARAM.LSTM_num_proj
        outputs = tf.reshape(outputs, [-1, rnn_output_num])
        in_size = rnn_output_num
      out_size = PARAM.OUTPUT_SIZE
      weights = tf.get_variable('weights1', [in_size, out_size],
                                initializer=tf.random_normal_initializer(stddev=0.01))
      biases = tf.get_variable('biases1', [out_size],
                               initializer=tf.constant_initializer(0.0))
      mask = tf.nn.relu(tf.matmul(outputs, weights) + biases)
      self._mask = tf.reshape(
          mask, [self.batch_size, -1, PARAM.OUTPUT_SIZE])

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if infer:
      if PARAM.DECODING_MASK_POSITION == 'mag':
        self._cleaned = rm_norm_mag_spec(self._mask*self._norm_x_mag_spec)
      elif PARAM.DECODING_MASK_POSITION == 'logmag':
        self._cleaned = rm_norm_logmag_spec(self._mask*self._norm_x_logmag_spec, self._log_bias)
      return

    if PARAM.TRAINING_MASK_POSITION == 'mag':
      self._cleaned = self._mask*self._norm_x_mag_spec
    elif PARAM.TRAINING_MASK_POSITION == 'logmag':
      self._cleaned = self._mask*self._norm_x_logmag_spec
    if PARAM.MASK_TYPE == 'PSIRM':
      self._labels *= tf.cos(theta_x_batch-theta_y_batch)

    if PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE:
      if PARAM.LABEL_TYPE == 'mag':
        self._cleaned = normedLogmag2normedMag(self._cleaned, self._log_bias)
      elif PARAM.LABEL_TYPE == 'logmag':
        self._cleaned = normedMag2normedLogmag(self._cleaned, self._log_bias)
    self._loss = PARAM.LOSS_FUNC(self._cleaned,self._labels)
    if tf.get_variable_scope().reuse:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      PARAM.CLIP_NORM)
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
    dims: [1]
    '''
    return self._real_logbias

  @property
  def cleaned(self):
    '''
    description: model outputs
    type: enhanced spectrum
    dims: [None,time,frequence]
    '''
    return self._cleaned

  @property
  def inputs(self):
    '''
    description: model inputs
    type: mixture spectrum
    dims: [None,time,frequence]
    '''
    return self._inputs

  @property
  def labels(self):
    '''
    description: trainning reference
    type: clean spectrum
    dims: [None,time,frequence]
    '''
    return self._labels

  @property
  def mask(self):
    '''
    description: wiener filtering mat
    type:
    dims: same to spectrum
    '''
    return self._mask

  @property
  def lengths(self):
    '''
    description: dynamic batch_size
    type: an int number
    dims: [1]
    '''
    return self._lengths

  @property
  def lr(self):
    '''
    description: learning rate
    type:
    dims: [1]
    '''
    return self._lr

  @property
  def loss(self):
    '''
    description: model loss
    type:
    dims: [1]
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



















