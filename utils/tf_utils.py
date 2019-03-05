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

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    sys.stdout.flush()
