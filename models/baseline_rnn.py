# speech enhancement

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


class Model_Baseline(object):
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
    self._log_bias = tf.get_variable('logbias', [1], trainable=FLAGS.PARAM.LOG_BIAS_TRAINABLE,
                                     initializer=tf.constant_initializer(FLAGS.PARAM.INIT_LOG_BIAS))
    self._real_logbias = self._log_bias + FLAGS.PARAM.MIN_LOG_BIAS
    self._x_mag_spec = x_mag_spec_batch
    self._norm_x_mag_spec = norm_mag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)
    self._norm_x_logmag_spec = norm_logmag_spec(self._x_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)

    self._y_mag_spec = y_mag_spec_batch
    self._norm_y_mag_spec = norm_mag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX)
    self._norm_y_logmag_spec = norm_logmag_spec(self._y_mag_spec, FLAGS.PARAM.MAG_NORM_MAX, self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)

    self._lengths = lengths_batch
    self._batch_size = tf.shape(self._lengths)[0]

    self._x_theta = theta_x_batch
    self._y_theta = theta_y_batch
    self._model_type = FLAGS.PARAM.MODEL_TYPE

    if FLAGS.PARAM.INPUT_TYPE == 'mag':
      self.net_input = self._norm_x_mag_spec
    elif FLAGS.PARAM.INPUT_TYPE == 'logmag':
      self.net_input = self._norm_x_logmag_spec
    if FLAGS.PARAM.LABEL_TYPE == 'mag':
      self._y_labels = self._norm_y_mag_spec
    elif FLAGS.PARAM.LABEL_TYPE == 'logmag':
      self._y_labels = self._norm_y_logmag_spec

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

    self.fw_final_state = fw_final_states
    self.bw_final_state = bw_final_states
    # print(fw_final_states[0][0].get_shape().as_list())

    # print(np.shape(fw_final_states),np.shape(bw_final_states))

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
                               initializer=tf.constant_initializer(FLAGS.PARAM.INIT_MASK_VAL))
    if FLAGS.PARAM.TIME_NOSOFTMAX_ATTENTION:
      with tf.variable_scope('fullconnectCoef'):
        weights_coef = tf.get_variable('weights_coef', [in_size, 1],
                                       initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01))
        biases_coef = tf.get_variable('biases_coef', [1],
                                      initializer=tf.constant_initializer(0.0))
      raw_mask = tf.reshape(tf.matmul(outputs, weights) + biases, [self._batch_size,-1,FLAGS.PARAM.OUTPUT_SIZE]) # [batch,time,fre]
      batch_coef_vec = tf.nn.relu(tf.reshape(tf.matmul(outputs, weights_coef) + biases_coef, [self._batch_size,-1])) # [batch, time]
      mask = tf.multiply(raw_mask,
                         tf.reshape(batch_coef_vec,[self._batch_size,-1,1]))
    else:
      if FLAGS.PARAM.POST_BN:
        linear_out = tf.matmul(outputs, weights)
        with tf.variable_scope('POST_Batch_Norm_Layer'):
          if_BRN = (FLAGS.PARAM.MVN_TYPE == 'BRN')
          if FLAGS.PARAM.SELF_BN:
            linear_out = tf.layers.batch_normalization(linear_out, training=True, renorm=if_BRN)
          else:
            linear_out = tf.layers.batch_normalization(linear_out,
                                                       training=(
                                                           behavior == self.train or behavior == self.validation),
                                                       renorm=if_BRN)
          weights2 = tf.get_variable('weights1', [out_size, out_size],
                                     initializer=tf.random_normal_initializer(stddev=0.01))
          biases2 = tf.get_variable('biases1', [out_size],
                                    initializer=tf.constant_initializer(FLAGS.PARAM.INIT_MASK_VAL))
          linear_out = tf.matmul(linear_out,weights2) + biases2
      else:
        linear_out = tf.matmul(outputs, weights) + biases
      mask = linear_out
      if FLAGS.PARAM.ReLU_MASK:
        mask = tf.nn.relu(linear_out)

    # endregion

    self._mask = tf.reshape(
        mask, [self._batch_size, -1, FLAGS.PARAM.OUTPUT_SIZE])

    if FLAGS.PARAM.TRAINING_MASK_POSITION == 'mag':
      self._y_estimation = self._mask*(self._norm_x_mag_spec+FLAGS.PARAM.SPEC_EST_BIAS)
    elif FLAGS.PARAM.TRAINING_MASK_POSITION == 'logmag':
      self._y_estimation = self._mask*(self._norm_x_logmag_spec+FLAGS.PARAM.SPEC_EST_BIAS)

    # region get infer spec
    if FLAGS.PARAM.DECODING_MASK_POSITION == 'mag':
      self._y_mag_estimation = rm_norm_mag_spec(self._mask*(self._norm_x_mag_spec+FLAGS.PARAM.SPEC_EST_BIAS),
                                                FLAGS.PARAM.MAG_NORM_MAX)
    elif FLAGS.PARAM.DECODING_MASK_POSITION == 'logmag':
      self._y_mag_estimation = rm_norm_logmag_spec(self._mask*(self._norm_x_logmag_spec+FLAGS.PARAM.SPEC_EST_BIAS),
                                                   FLAGS.PARAM.MAG_NORM_MAX,
                                                   self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    '''
    _y_mag_estimation is estimated mag_spec
    _y_estimation is loss_targe, mag_sepec or logmag_spec
    '''
    # endregion

    # region prepare y_estimation
    if FLAGS.PARAM.TRAINING_MASK_POSITION != FLAGS.PARAM.LABEL_TYPE:
      if FLAGS.PARAM.LABEL_TYPE == 'mag':
        self._y_estimation = normedLogmag2normedMag(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                    self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
      elif FLAGS.PARAM.LABEL_TYPE == 'logmag':
        self._y_estimation = normedMag2normedLogmag(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                    self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    # endregion

    # region CBHG
    if FLAGS.PARAM.USE_CBHG_POST_PROCESSING:
      cbhg_kernels = 8 # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act as "K-grams"
      cbhg_conv_channels = 128 # Channels of the convolution bank
      cbhg_pool_size = 2 # pooling size of the CBHG
      cbhg_projection = 256 # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
      cbhg_projection_kernel_size = 3 # kernel_size of the CBHG projections
      cbhg_highwaynet_layers = 4 # Number of HighwayNet layers
      cbhg_highway_units = 128 # Number of units used in HighwayNet fully connected layers
      cbhg_rnn_units = 128 # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in shape
      batch_norm_position = 'before'
      # is_training = True
      is_training = bool(behavior == self.train)
      post_cbhg = CBHG(cbhg_kernels, cbhg_conv_channels, cbhg_pool_size, [cbhg_projection, FLAGS.PARAM.OUTPUT_SIZE],
                       cbhg_projection_kernel_size, cbhg_highwaynet_layers,
                       cbhg_highway_units, cbhg_rnn_units, batch_norm_position, is_training, name='CBHG_postnet')

      #[batch_size, decoder_steps(mel_frames), cbhg_channels]
      self._cbhg_inputs_y_est = self._y_estimation
      cbhg_outputs = post_cbhg(self._y_estimation, None)

      frame_projector = FrameProjection(FLAGS.PARAM.OUTPUT_SIZE,scope='CBHG_proj_to_spec')
      self._y_estimation = frame_projector(cbhg_outputs)

      if FLAGS.PARAM.DECODING_MASK_POSITION != FLAGS.PARAM.TRAINING_MASK_POSITION:
        print('DECODING_MASK_POSITION must be equal to TRAINING_MASK_POSITION when use CBHG post processing.')
        exit(-1)
      if FLAGS.PARAM.DECODING_MASK_POSITION == 'mag':
        self._y_mag_estimation = rm_norm_mag_spec(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX)
      elif FLAGS.PARAM.DECODING_MASK_POSITION == 'logmag':
        self._y_mag_estimation = rm_norm_logmag_spec(self._y_estimation, FLAGS.PARAM.MAG_NORM_MAX,
                                                     self._log_bias, FLAGS.PARAM.MIN_LOG_BIAS)
    # endregion


    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if behavior == self.infer:
      return

    # region get labels LOSS
    # Labels
    if FLAGS.PARAM.MASK_TYPE == 'PSM':
      self._y_labels *= tf.cos(self._x_theta-self._y_theta)
    elif FLAGS.PARAM.MASK_TYPE == 'fixPSM':
      self._y_labels *= (1.0+tf.cos(self._x_theta-self._y_theta))*0.5
    elif FLAGS.PARAM.MASK_TYPE == 'AcutePM':
      self._y_labels *= tf.nn.relu(tf.cos(self._x_theta-self._y_theta))
    elif FLAGS.PARAM.MASK_TYPE == 'PowFixPSM':
      self._y_labels *= tf.pow(tf.abs((1.0+tf.cos(self._x_theta-self._y_theta))*0.5),
                               FLAGS.PARAM.POW_FIX_PSM_COEF)
    elif FLAGS.PARAM.MASK_TYPE == 'IRM':
      pass
    else:
      tf.logging.error('Mask type error.')
      exit(-1)

    # LOSS
    if FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'SPEC_MSE': # log_mag and mag MSE
      self._loss = loss.reduce_sum_frame_batchsize_MSE(self._y_estimation,self._y_labels)
      if FLAGS.PARAM.USE_CBHG_POST_PROCESSING:
        if FLAGS.PARAM.DOUBLE_LOSS:
          self._loss = FLAGS.PARAM.CBHG_LOSS_COEF1*loss.reduce_sum_frame_batchsize_MSE(
            self._cbhg_inputs_y_est,self._y_labels) + FLAGS.PARAM.CBHG_LOSS_COEF2*self._loss
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
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "RELATED_MSE":
      self._loss = loss.relative_reduce_sum_frame_batchsize_MSE(self._y_estimation,self._y_labels,FLAGS.PARAM.RELATED_MSE_IGNORE_TH)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE(
          self._y_estimation, self._y_labels, FLAGS.PARAM.AUTO_RELATED_MSE_AXIS_FIT_DEG)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE2":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v2(
          self._y_estimation, self._y_labels,
          FLAGS.PARAM.AUTO_RELATED_MSE_AXIS_FIT_DEG,
          FLAGS.PARAM.LINEAR_BROKER,)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE3":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v3(
          self._y_estimation, self._y_labels,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS3_A, FLAGS.PARAM.AUTO_RELATIVE_LOSS3_B,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS3_C1, FLAGS.PARAM.AUTO_RELATIVE_LOSS3_C2)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE4":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v4(
          self._y_estimation, self._y_labels, FLAGS.PARAM.AUTO_RELATED_MSE_AXIS_FIT_DEG)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE5":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v5(
          self._y_estimation, self._y_labels)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE6":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v6(
          self._y_estimation, self._y_labels,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS6_A, FLAGS.PARAM.AUTO_RELATIVE_LOSS6_B,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS6_C1, FLAGS.PARAM.AUTO_RELATIVE_LOSS6_C2)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE7":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v7(
          self._y_estimation, self._y_labels,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS7_A1, FLAGS.PARAM.AUTO_RELATIVE_LOSS7_A2,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS7_B, FLAGS.PARAM.AUTO_RELATIVE_LOSS7_C1, FLAGS.PARAM.AUTO_RELATIVE_LOSS7_C2)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE8":
      self._loss = loss.auto_ingore_relative_reduce_sum_frame_batchsize_MSE_v8(
          self._y_estimation, self._y_labels,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS8_A, FLAGS.PARAM.AUTO_RELATIVE_LOSS8_B,
          FLAGS.PARAM.AUTO_RELATIVE_LOSS8_C1, FLAGS.PARAM.AUTO_RELATIVE_LOSS8_C2)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == "AUTO_RELATED_MSE_USE_COS":
      self._loss = loss.cos_auto_ingore_relative_reduce_sum_frame_batchsize_MSE(self._y_estimation,self._y_labels,
                                                                                FLAGS.PARAM.COS_AUTO_RELATED_MSE_W)
    elif FLAGS.PARAM.LOSS_FUNC_FOR_MAG_SPEC == 'MEL_AUTO_RELATED_MSE':
      # type(y_estimation) = FLAGS.PARAM.LABEL_TYPE
      self._loss = loss.MEL_AUTO_RELATIVE_MSE(self._y_estimation, self._norm_y_mag_spec,
                                              FLAGS.PARAM.MEL_NUM, FLAGS.PARAM.AUTO_RELATED_MSE_AXIS_FIT_DEG)
    else:
      print('Loss type error.')
      exit(-1)
    # endregion

    if behavior == self.validation:
      '''
      val model cannot train.
      '''
      return
    self._lr = tf.Variable(0.0, trainable=False) #TODO
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
    dims: [0]
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
