import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
CPU = '/cpu:0'


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
def norm_logmag_spec(mag_spec, MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS+MAG_NORM_MAX) / tf.log(10.0)

  # mag_spec = tf.clip_by_value(mag_spec, 0, MAG_NORM_MAX)
  logmag_spec = tf.log(mag_spec+tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS)/tf.log(10.0)
  logmag_spec -= LOG_NORM_MIN
  normed_logmag = logmag_spec / (LOG_NORM_MAX - LOG_NORM_MIN)
  return normed_logmag

# Inverse process of norm_mag_spec()
def rm_norm_mag_spec(normed_mag, MAG_NORM_MAX):
  mag_spec = normed_mag * (MAG_NORM_MAX - 0)
  return mag_spec

# Inverse process of norm_logmag_spec()
def rm_norm_logmag_spec(normed_logmag, MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS):
  LOG_NORM_MIN = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS) / tf.log(10.0)
  LOG_NORM_MAX = tf.log(tf.nn.relu(log_bias)+DEFAULT_LOG_BIAS+MAG_NORM_MAX) / tf.log(10.0)

  mag_spec = normed_logmag * (LOG_NORM_MAX - LOG_NORM_MIN)
  mag_spec += LOG_NORM_MIN
  mag_spec *= tf.log(10.0)
  mag_spec = tf.exp(mag_spec) - DEFAULT_LOG_BIAS - tf.nn.relu(log_bias)
  return mag_spec

#
def normedMag2normedLogmag(normed_mag, MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS):
  return norm_logmag_spec(rm_norm_mag_spec(normed_mag, MAG_NORM_MAX), MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS)

#
def normedLogmag2normedMag(normed_logmag, MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS):
  return norm_mag_spec(rm_norm_logmag_spec(normed_logmag, MAG_NORM_MAX, log_bias, DEFAULT_LOG_BIAS), MAG_NORM_MAX)

def melspec_form_realStft(mag_spectrograms, sample_rate, num_mel_bins):
  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = mag_spectrograms.shape[-1].value
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

  return mel_spectrograms

def mfccs_form_realStft(mag_spectrograms, sample_rate, num_mel_bins, n_mfccs):
  mel_spectrograms = melspec_form_realStft(mag_spectrograms, sample_rate, num_mel_bins)

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :n_mfccs]
  return mfccs
