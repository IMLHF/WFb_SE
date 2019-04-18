import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
CPU = '/cpu:0'


class HighwayNet:
  def __init__(self, units, name=None):
    # self.input_units = input_units
    self.units = units
    self.scope = 'HighwayNet' if name is None else name

    # with tf.variable_scope(self.scope):
    #   self.H_w = tf.get_variable('weights_H', [self.input_units, self.units],
    #                              initializer=tf.random_normal_initializer(stddev=0.01))
    #   self.H_b = tf.get_variable('biases_H', [self.units],
    #                              initializer=tf.random_normal_initializer(stddev=0.01))
    #   self.T_w = tf.get_variable('weights_T', [self.input_units, self.units],
    #                              initializer=tf.random_normal_initializer(stddev=0.01))
    #   self.T_b = tf.get_variable('biases_T', [self.units],
    #                              initializer=tf.constant_initializer(-1.))
    self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name='H')
    self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      H = self.H_layer(inputs)
      T = self.T_layer(inputs)
      # H = tf.nn.relu(tf.matmul(inputs,self.H_w)+self.H_b)
      # T = tf.nn.sigmoid(tf.matmul(inputs,self.T_w)+self.T_b)
      return H * T + inputs * (1. - T)


class CBHG:
  def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, bnorm, is_training, name=None):
    self.K = K
    self.conv_channels = conv_channels
    self.pool_size = pool_size

    self.projections = projections
    self.projection_kernel_size = projection_kernel_size
    self.bnorm = bnorm

    self.is_training = is_training
    self.scope = 'CBHG' if name is None else name

    self.highway_units = highway_units
    self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}_'.format(self.scope, i+1)) for i in range(n_highwaynet_layers)]
    self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_forward_RNN'.format(self.scope))
    self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_backward_RNN'.format(self.scope))

  def __call__(self, inputs, input_lengths):
    with tf.variable_scope(self.scope):
      with tf.variable_scope('conv_bank'):
        #Convolution bank: concatenate on the last axis to stack channels from all convolutions
        #The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
        #This makes one of the strengths of the CBHG block on sequences.
        conv_outputs = tf.concat(
          [conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training, 0.2, self.bnorm, 'conv1d_{}_'.format(k)) for k in range(1, self.K+1)],
          axis=-1
          )

      # Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
      maxpool_output = tf.layers.max_pooling1d(
        conv_outputs,
        pool_size=self.pool_size,
        strides=1,
        padding='same')

      # Two projection layers
      proj1_output = conv1d(maxpool_output, self.projection_kernel_size, self.projections[0], tf.nn.relu, self.is_training, 0.2, self.bnorm, 'proj1')
      proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections[1], lambda _: _, self.is_training, 0.2, self.bnorm, 'proj2')

      #Residual connection
      highway_input = proj2_output + inputs

      #Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
      if highway_input.shape[2] != self.highway_units:
        highway_input = tf.layers.dense(highway_input, self.highway_units)

      #4-layer HighwayNet
      for highwaynet in self.highwaynet_layers:
        highway_input = highwaynet(highway_input)
      rnn_input = highway_input

      #Bidirectional RNN
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
        self._fw_cell,
        self._bw_cell,
        rnn_input,
        sequence_length=input_lengths,
        dtype=tf.float32)
      return tf.concat(outputs, axis=-1) # Concat forward and backward outputs


class FrameProjection:
  """Projection layer to r * num_mels dimensions or num_mels dimensions
  """
  def __init__(self, shape, activation=None, scope=None):
    """
    Args:
      shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
      activation: callable, activation function
      scope: FrameProjection scope.
    """
    super(FrameProjection, self).__init__()

    self.shape = shape
    self.activation = activation

    self.scope = 'Linear_projection' if scope is None else scope
    self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))
    # with tf.variable_scope(self.scope):
    #   self._w = tf.get_variable('weights', [self.input_units, self.units],
    #                             initializer=tf.random_normal_initializer(stddev=0.01))
    #   self._b = tf.get_variable('biases', [self.units],
    #                             initializer=tf.random_normal_initializer(stddev=0.01))

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      #If activation==None, this returns a simple Linear projection
      #else the projection will be passed through an activation function
      # output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
      #   name='projection_{}'.format(self.scope))
      output = self.dense(inputs)
      # output = tf.matmul(inputs,self._w)+self._b
      # if self.activation is not None:
      #   output = self.activation(output)
      return output


def new_variable_xavier_L2regular(name, shape, weight_decay=0.001,
                                  init=tf.contrib.layers.xavier_initializer()):
  with tf.device(CPU):
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    var = tf.get_variable(name, shape=shape, initializer=init,
                          regularizer=regularizer)
  return var


def new_variable(name, shape,
                 init=tf.random_normal_initializer()):
  with tf.device(CPU):
    var = tf.get_variable(name, shape=shape, initializer=init)
  return var


def lstm_cell(n_units, n_proj, activation_fun):
  return tf.contrib.rnn.LSTMCell(
      n_units, forget_bias=1.0, use_peepholes=True,
      num_proj=n_proj,
      initializer=tf.contrib.layers.xavier_initializer(),
      state_is_tuple=True, activation=activation_fun)


def GRU_cell(n_units, activation_fun):
  return tf.contrib.rnn.GRUCell(
      n_units,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      activation=activation_fun)


def tdnn_layer_not_active(x, time_width, time_stride, units_num, padding, name):
  shape = (time_width, np.shape(x)[-1], units_num)
  weights = new_variable(shape=shape, name=name+'_weight')
  bias = new_variable(shape=(units_num), name=name+'_bias')
  return tf.add(tf.nn.conv1d(x,
                             weights,
                             stride=time_stride,
                             padding=padding,
                             name=name + "_output"),
                bias)


def relu_tdnn_layer(x, time_width, time_stride, units_num, padding, name):
  return tf.nn.relu(tdnn_layer_not_active(x, time_width, time_stride, units_num, padding, name))


def sigmoid_tdnn_layer(x, time_width, time_stride, units_num, padding, name):
  return tf.nn.sigmoid(tdnn_layer_not_active(x, time_width, time_stride, units_num, padding, name))


def tanh_tdnn_layer(x, time_width, time_stride, units_num, padding, name):
  return tf.nn.tanh(tdnn_layer_not_active(x, time_width, time_stride, units_num, padding, name))


def elu_tdnn_layer(x, time_width, time_stride, units_num, padding, name):
  return tf.nn.elu(tdnn_layer_not_active(x, time_width, time_stride, units_num, padding, name))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_to_collection(**kwargs):
  for key in kwargs.keys():
    tf.add_to_collection(key, kwargs[key])

def get_gpu_batch_size_list(n_x,n_gpu):
  gpu_batch_size=n_x//n_gpu
  gpu_batch_size_list=tf.constant([],dtype=np.int32)
  for i in range(n_gpu-1):
    n_x-=gpu_batch_size
    gpu_batch_size_list=tf.concat([gpu_batch_size_list,[gpu_batch_size]],0)
  gpu_batch_size_list=tf.concat([gpu_batch_size_list,[n_x]],0)
  return gpu_batch_size_list

def show_variables(vars_):
    slim.model_analyzer.analyze_vars(vars_, print_info=True)
    sys.stdout.flush()


def show_all_variables():
    model_vars = tf.trainable_variables()
    show_variables(model_vars)


# norm mag to [0,1]
def norm_mag_spec(mag_spec, MAG_NORM_MAX):
  # mag_spec = tf.clip_by_value(mag_spec, 0, MAG_NORM_MAX)
  normed_mag = mag_spec / (MAG_NORM_MAX - 0)
  return normed_mag

# add bias and logarithm to mag, dispersion to [0,1]
def norm_logmag_spec(mag_spec, MAG_NORM_MAX, log_bias, MIN_LOG_BIAS):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+MIN_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+MIN_LOG_BIAS+MAG_NORM_MAX) / tf.log(10.0)

  # mag_spec = tf.clip_by_value(mag_spec, 0, MAG_NORM_MAX)
  logmag_spec = tf.log(mag_spec+tf.nn.relu(log_bias)+MIN_LOG_BIAS)/tf.log(10.0)
  logmag_spec -= LOG_NORM_MIN
  normed_logmag = logmag_spec / (LOG_NORM_MAX - LOG_NORM_MIN)
  return normed_logmag

# Inverse process of norm_mag_spec()
def rm_norm_mag_spec(normed_mag, MAG_NORM_MAX):
  mag_spec = normed_mag * (MAG_NORM_MAX - 0)
  return mag_spec

# Inverse process of norm_logmag_spec()
def rm_norm_logmag_spec(normed_logmag, MAG_NORM_MAX, log_bias, MIN_LOG_BIAS):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+MIN_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+MIN_LOG_BIAS+MAG_NORM_MAX) / tf.log(10.0)

  mag_spec = normed_logmag * (LOG_NORM_MAX - LOG_NORM_MIN)
  mag_spec += LOG_NORM_MIN
  mag_spec *= tf.log(10.0)
  mag_spec = tf.exp(mag_spec) - MIN_LOG_BIAS - tf.nn.relu(log_bias)
  return mag_spec

#
def normedMag2normedLogmag(normed_mag, MAG_NORM_MAX, log_bias, MIN_LOG_BIAS):
  return norm_logmag_spec(rm_norm_mag_spec(normed_mag, MAG_NORM_MAX), MAG_NORM_MAX, log_bias, MIN_LOG_BIAS)

#
def normedLogmag2normedMag(normed_logmag, MAG_NORM_MAX, log_bias, MIN_LOG_BIAS):
  return norm_mag_spec(rm_norm_logmag_spec(normed_logmag, MAG_NORM_MAX, log_bias, MIN_LOG_BIAS), MAG_NORM_MAX)

def melspec_form_realStft(mag_spectrograms, sample_rate, num_mel_bins):
  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = mag_spectrograms.shape[-1].value
  # num_spectrogram_bins = mag_spectrograms.shape[-1]
  lower_edge_hertz, upper_edge_hertz = sample_rate*0.015625, sample_rate*0.475
  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                              num_spectrogram_bins,
                                                                              sample_rate,
                                                                              lower_edge_hertz,
                                                                              upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    mag_spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(mag_spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
  # mel_spectrograms = tf.reshape(mel_spectrograms, tf.concat([mag_spectrograms.shape[:-1],
  #                                                            linear_to_mel_weight_matrix.shape[-1:]], axis=-1))

  return mel_spectrograms

def mfccs_form_realStft(mag_spectrograms, sample_rate, num_mel_bins, n_mfccs):
  mel_spectrograms = melspec_form_realStft(mag_spectrograms, sample_rate, num_mel_bins)

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :n_mfccs]
  return mfccs


def sum_attention_with_final_state(features, state, final_dim, units):
  '''
  features:
    dim: [batch,time,final_dim]
  '''
  with tf.variable_scope('sum_attention_fc'):
    # attention layer
    with tf.variable_scope('attention_scorer'):
      score_features_hidden_W = tf.layers.Dense(units)
      score_state_hidden_W = tf.layers.Dense(units)
      score_vec_W = tf.layers.Dense(1)
  state_with_time_axis = tf.expand_dims(state, -2) # [batch,1,final_dim]
  score = tf.nn.tanh(score_features_hidden_W(features) + score_state_hidden_W(state_with_time_axis)) # [batch,time,units]
  attention_weights = tf.nn.softmax(score_vec_W(score), axis=-2) # [batch,time,1]
  context_vector = tf.multiply(attention_weights, features)
  context_vector = tf.reduce_sum(context_vector, axis=-2)
  return context_vector


def sum_attention(features, final_dim, units):
  '''
  features:
    dim: [batch,time,final_dim]
  '''
  with tf.variable_scope('sum_attention_fc'):
    # attention layer
    with tf.variable_scope('attention_scorer'):
      score_features_hidden_W = tf.layers.Dense(units)
      score_vec_W = tf.layers.Dense(1)
  score = tf.nn.tanh(score_features_hidden_W(features)) # [batch,time,units]
  attention_weights = tf.nn.softmax(score_vec_W(score), axis=-2) # [batch,time,1]
  context_vector = tf.multiply(attention_weights, features)
  context_vector = tf.reduce_sum(context_vector, axis=-2)
  return context_vector


def sum_attention_v2(inputs, batch, input_finnal_dim):
  '''
  inputs: func inputs of calculate audio-suitable log_bias
    dims" [batch,time,fea_dim]
  '''
  fea_dim = input_finnal_dim
  with tf.variable_scope('sum_attention_fc'):
    # attention layer
    with tf.variable_scope('attention_scorer'):
      weights_scorer = tf.get_variable('weights_scorer', [fea_dim, 1],
                                       initializer=tf.random_normal_initializer(stddev=0.01))
      biases_scorer = tf.get_variable('biases_scorer', [1],
                                      initializer=tf.constant_initializer(0.0))
      attention_alpha_vec = tf.matmul(tf.reshape(inputs, [-1, input_finnal_dim]),
                                      weights_scorer) + biases_scorer  # [batch*time,1]
      attention_alpha_vec = tf.reshape(attention_alpha_vec,[batch,1,-1]) # [batch,1,time]
      attention_alpha_vec = tf.nn.softmax(attention_alpha_vec, axis=-1)
      attened_vec = tf.reshape(tf.matmul(attention_alpha_vec, inputs), [batch,-1])# [batch,fea_dim]
      return attened_vec


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, bnorm, scope):
  assert bnorm in ('before', 'after')
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation if bnorm == 'after' else None,
      padding='SAME')
    batched = conv1d_output
    # batched = tf.layers.batch_normalization(conv1d_output, training=True)
    activated = activation(batched) if bnorm == 'before' else batched
    return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
                             name='dropout_{}'.format(scope))
