import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
  '''Sinusoidal Positional_Encoding. See 3.5 in paper "Attention is all your need."
  inputs: 3d tensor. (N, T, E)
  maxlen: scalar. Must be >= T
  masking: Boolean. If True, padding positions are set to zeros.
  scope: Optional scope for `variable_scope`.

  returns
  3d tensor that has the same shape as inputs.
  '''

  E = inputs.get_shape().as_list()[-1]  # static
  N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # position indices
    position_ind = tf.tile(tf.expand_dims(
        tf.range(T), 0), [N, 1])  # (N, T)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i-i % 2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = tf.convert_to_tensor(
        position_enc, tf.float32)  # (maxlen, E)

    # lookup
    outputs = tf.nn.embedding_lookup(position_enc, position_ind)

    # masks
    if masking:
        outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

    return tf.to_float(outputs)


def attention_score_mask(scores, KV_lengths, mask_value=None):
  """
  Args:
    scores: [batch, time_query, time_kv], src_seq_length[0] is true length of scores[0, *]
    KV_lengths: [batch,], keys and values lengths.
  Return:
    masked_scores: [batch, time_query, timekv]
  Others:
    mask before softmax.
  """
  if mask_value is None:
    # mask_value = dtypes.as_dtype(scores.dtype).as_numpy_dtype(-np.inf)
    mask_value = -2 ** 32 + 1.0
  time_kv = tf.shape(scores)[2]
  mask = tf.sequence_mask(KV_lengths, maxlen=time_kv) # [batch, time_kv]
  mask = tf.expand_dims(mask, 1) # [batch, 1, time_kv]
  mask = tf.tile(mask, [1, tf.shape(scores)[1], 1]) # [batch, time_query, time_kv]
  score_mask_values = mask_value * tf.ones_like(scores)
  return tf.where(mask, scores, score_mask_values)


def causality_mask_for_self_attention(inputs, mask_value=None):
  """
  mask before softmax.
  """
  if mask_value is None:
    # mask_value = dtypes.as_dtype(inputs.dtype).as_numpy_dtype(-np.inf)
    mask_value = -2 ** 32 + 1.0
  diag_vals = tf.ones_like(inputs[0, :, :])  # (time_query, time_kv), always have "time_query == time_kv"
  tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (time_query, time_kv)
  masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (batch, time_query, time_kv)

  paddings = tf.ones_like(masks) * mask_value
  outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
  return outputs


def query_time_mask_for_train(inputs, query_lengths):
  """
  Args:
    inputs: [batch, time_query, time_kv], src_seq_length[0] is true length of scores[0, *]
    query_lengths: [batch,]
  Return:
    masked_scores: [batch, time_query, time_kv]
  Others:
    mask after softmax. before is ok so.
    same to rnn_seq_lengths, no use for inference.
    action as sequence_mask.
  """
  # if mask_value is None:
  #   mask_value = 0.0
  time_query = tf.shape(inputs)[1]
  mask = tf.sequence_mask(query_lengths, maxlen=time_query, dtype=inputs.dtype) # [batch, time_query]
  mask = tf.expand_dims(mask, 2) # [batch, time_query, 1]
  mask = tf.tile(mask, [1, 1, tf.shape(inputs)[2]]) # [batch, time_query, time_kv]
  # score_mask_values = mask_value * tf.ones_like(inputs)
  # return tf.where(mask, inputs, score_mask_values)
  outputs = tf.multiply(inputs, mask)
  return outputs


def scaled_dot_product_attention(Q, K, V, KV_lengths, Q_lengths=None,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
  '''See 3.2.1.
  Q: Packed queries. 3d tensor. [N, T_q, d_k].
  K: Packed keys. 3d tensor. [N, T_k, d_k].
  V: Packed values. 3d tensor. [N, T_k, d_v].
  causality: If True, applies masking for future blinding
  dropout_rate: A floating point number of [0, 1].
  training: boolean for controlling droput
  scope: Optional scope for `variable_scope`.
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    #
    K_T = tf.transpose(K, [0, 2, 1])
    outputs = tf.matmul(Q, K_T)  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # attention_score_mask
    outputs = attention_score_mask(outputs, KV_lengths)


    # causality or future blinding masking
    if causality:
      outputs = causality_mask_for_self_attention(outputs)

    # softmax
    outputs = tf.nn.softmax(outputs)
    attention = tf.transpose(outputs, [0, 2, 1])
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # query masking
    # remove query_time_mask_for_train and add tf.sequence_mask at calculate loss ?
    # if not PARAM.rm_query_mask:
    #   outputs = query_time_mask_for_train(outputs, Q_lengths)

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

  return outputs


def layer_norm(inputs, epsilon=1e-8, scope="layer_norm"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries, keys, values,
                        d_model, KV_lengths, Q_lengths,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
  '''Applies multihead attention. See 3.2.2
  queries: A 3d tensor with shape of [N, T_q, d_model].
  keys: A 3d tensor with shape of [N, T_k, d_model].
  values: A 3d tensor with shape of [N, T_k, d_model].
  num_heads: An int. Number of heads.
  dropout_rate: A floating point number.
  training: Boolean. Controller of mechanism for dropout.
  causality: Boolean. If true, units that reference the future are masked.
  scope: Optional scope for `variable_scope`.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False) # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False) # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False) # (N, T_k, d_model)

    # Split and concat
    assert d_model % num_heads == 0, "d_model % num_heads == 0 is required. d_model:%d, num_heads:%d." % (
        d_model, num_heads)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

    # Attention
    KV_lengths = tf.tile(KV_lengths, [num_heads])
    Q_lengths = tf.tile(Q_lengths, [num_heads])
    # print("QKV", Q_.get_shape().as_list(), K_.get_shape().as_list(), V_.get_shape().as_list(),)
    outputs = scaled_dot_product_attention(Q_, K_, V_, KV_lengths, Q_lengths,
                                           causality, dropout_rate, training)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = layer_norm(outputs)

  return outputs


def positionwise_FC(inputs, num_units, scope="positionwise_feedforward"):
  '''position-wise feed forward net. See 3.3

  inputs: A 3d tensor with shape of [N, T, C].
  num_units: A list of two integers.
  scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # Inner layer
    outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

    # Outer layer
    outputs = tf.layers.dense(outputs, num_units[1])

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = layer_norm(outputs)

  return outputs

