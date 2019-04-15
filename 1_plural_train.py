'''
for :
   models.plural_mask_model
'''


import time
import tensorflow as tf
import numpy as np
import sys
import utils
import utils.audio_tool
import os
import shutil
import wave
import gc
from FLAGS import PARAM
from tensorflow.python import debug as tf_debug
from dataManager.mixed_aishell_8k_tfrecord_io import generate_tfrecord, get_batch_use_tfdata



def train_one_epoch(sess, tr_model):
  """Runs the model one epoch on given data."""
  tr_loss_mag, tr_loss_phase, tr_loss, i = 0, 0, 0, 0
  stime = time.time()
  while True:
    try:
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')
      _, loss, mag_loss, phase_loss, current_batchsize, model_lr = sess.run(
          # [tr_model.train_op, tr_model.loss, tf.shape(tr_model.lengths)[0]])
          [tr_model.train_op,
           tr_model.loss,
           tr_model.mag_loss,
           tr_model.phase_loss,
           tr_model.batch_size,
           tr_model.lr,
           #tr_model.threshold
           ])
      tr_loss_mag += mag_loss
      tr_loss_phase += phase_loss
      tr_loss += loss
      # print(loss,flush=True)
      # print(np.min(log_bias),np.max(log_bias),np.mean(log_bias),np.sqrt(np.var(log_bias)),flush=True)
      # print(np.min(threshold),np.max(threshold),np.mean(threshold),np.sqrt(np.var(threshold)),flush=True)
      # print("\r    Batch loss: %f" % loss, end="")
      if (i+1) % PARAM.minibatch_size == 0:
        lr = sess.run(tr_model.lr)
        costtime = time.time()-stime
        stime = time.time()
        avg_loss = tr_loss / (i+1)
        avg_loss_mag = tr_loss_mag / (i+1)
        avg_loss_phase = tr_loss_phase / (i+1)
        train_epoch_msg = ("MINIBATCH %05d: AVG.LOSS (%04.4f, %04.4f, %04.4f), "
                           "LR %02.6f, DURATION %06dS") % (
                               i + 1, avg_loss_mag, avg_loss_phase, avg_loss, lr, costtime)
        tf.logging.info(train_epoch_msg)
        with open(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log'), 'a+') as f:
          f.writelines(train_epoch_msg+'\n')
        sys.stdout.flush()
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss_mag /= i
  tr_loss_phase /= i
  tr_loss /= i
  return tr_loss_mag, tr_loss_phase, tr_loss, model_lr


def eval_one_epoch(sess, val_model):
  """Cross validate the model on given data."""
  val_loss_mag, val_loss_phase, val_loss = 0, 0, 0
  data_len = 0
  while True:
    try:
      loss, mag_loss, phase_loss, current_batchsize = sess.run([
          val_model.loss, val_model.mag_loss, val_model.phase_loss, val_model.batch_size
          # val_model.threshold,
          # val_model._loss1, val_model._loss2,
      ])
      # print(inputs)
      # exit(0)
      # print(loss2,'/',loss1)
      val_loss_mag += mag_loss
      val_loss_phase += phase_loss
      val_loss += loss
      # print(loss)
      # print(np.min(log_bias),np.max(log_bias),np.mean(log_bias),np.sqrt(np.var(log_bias)),flush=True)
      # print(np.min(threshold),np.max(threshold),np.mean(threshold),np.sqrt(np.var(threshold)),flush=True)
      # print("\r    Batch loss: %f" % loss, end="")
      sys.stdout.flush()
      data_len += 1
    except tf.errors.OutOfRangeError:
      break
  val_loss_mag /= data_len
  val_loss_phase /= data_len
  val_loss /= data_len
  return val_loss_mag, val_loss_phase, val_loss


def train():

  g = tf.Graph()
  with g.as_default():
    # region TFRecord+DataSet
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        train_tfrecords, val_tfrecords, _, _ = generate_tfrecord(
            gen=PARAM.GENERATE_TFRECORD)
        if PARAM.GENERATE_TFRECORD:
          tf.logging.info("TFRecords preparation over.")
          # exit(0)  # set gen=True and exit to generate tfrecords

        PSIRM = True if PARAM.PIPLINE_GET_THETA else False
        x_batch_tr, y_batch_tr, Xtheta_batch_tr, Ytheta_batch_tr, lengths_batch_tr, iter_train = get_batch_use_tfdata(
            train_tfrecords,
            get_theta=PSIRM)
        x_batch_val, y_batch_val,  Xtheta_batch_val, Ytheta_batch_val, lengths_batch_val, iter_val = get_batch_use_tfdata(
            val_tfrecords,
            get_theta=PSIRM)
    # endregion

    # build model
    with tf.name_scope('model'):
      tr_model = PARAM.SE_MODEL(x_batch_tr,
                                lengths_batch_tr,
                                y_batch_tr,
                                Xtheta_batch_tr,
                                Ytheta_batch_tr,
                                PARAM.SE_MODEL.train)
      tf.get_variable_scope().reuse_variables()
      val_model = PARAM.SE_MODEL(x_batch_val,
                                 lengths_batch_val,
                                 y_batch_val,
                                 Xtheta_batch_val,
                                 Ytheta_batch_val,
                                 PARAM.SE_MODEL.validation)

    utils.tf_tool.show_all_variables()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
    config.allow_soft_placement = False
    sess = tf.Session(config=config)
    sess.run(init)

    # region resume training
    if PARAM.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT))
      if ckpt and ckpt.model_checkpoint_path:
        # tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
      with open(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log'), 'a+') as f:
        f.writelines('Training resumed.\n')
    else:
      if os.path.exists(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log')):
        os.remove(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log'))
    # endregion

    # region validation before training.
    valstart_time = time.time()
    sess.run(iter_val.initializer)
    loss_mag_prev, loss_phase_prev, loss_prev = eval_one_epoch(sess,
                                                               val_model)
    cross_val_msg = "CROSSVAL PRERUN AVG.LOSS (%.4F, %.4F, %.4F)  costime %dS" % (
        loss_mag_prev, loss_phase_prev, loss_prev, time.time()-valstart_time)
    tf.logging.info(cross_val_msg)
    with open(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log'), 'a+') as f:
      f.writelines(cross_val_msg+'\n')

    tr_model.assign_lr(sess, PARAM.learning_rate)
    g.finalize()
    # endregion

    # epochs training
    for epoch in range(PARAM.start_epoch, PARAM.max_epochs):
      sess.run([iter_train.initializer, iter_val.initializer])
      start_time = time.time()

      # train one epoch
      tr_loss_mag, tr_loss_phase, tr_loss, model_lr = train_one_epoch(sess,
                                                                      tr_model)

      # Validation
      val_loss_mag, val_loss_phase, val_loss = eval_one_epoch(sess,
                                                              val_model)

      end_time = time.time()

      # Determine checkpoint path
      ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_%.4f_%.4f_cvloss%.4f_%.4f_%.4f_duration%ds" % (
          epoch + 1, model_lr, tr_loss_mag, tr_loss_phase, tr_loss, val_loss_mag, val_loss_phase, val_loss, end_time - start_time)
      ckpt_dir = os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT)
      if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
      ckpt_path = os.path.join(ckpt_dir, ckpt_name)

      # Relative loss between previous and current val_loss
      rel_impr_mag = (loss_mag_prev - val_loss_mag) / loss_mag_prev
      rel_impr_phase = (loss_phase_prev - val_loss_phase) / loss_phase_prev
      # Accept or reject new parameters
      msg = ""
      if (val_loss_mag < loss_mag_prev and val_loss_phase < loss_phase_prev) or (rel_impr_mag+rel_impr_phase)>0.0:
        tr_model.saver.save(sess, ckpt_path)
        # Logging train loss along with validation loss
        loss_prev = val_loss
        loss_mag_prev = val_loss_mag
        loss_phase_prev = val_loss_phase
        best_path = ckpt_path
        msg = ("Train Iteration %03d: \n"
               "    Train.LOSS (%.4f, %.4f, %.4f), lrate %e, Val.LOSS (%.4f, %.4f, %.4f),\n"
               "    %s, ckpt(%s) saved,\n"
               "    EPOCH DURATION: %.2fs\n") % (
            epoch + 1,
            tr_loss_mag, tr_loss_phase, tr_loss, model_lr, val_loss_mag, val_loss_phase, val_loss,
            "NNET Accepted", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      else:
        tr_model.saver.restore(sess, best_path)
        msg = ("Train Iteration %03d: \n"
               "    Train.LOSS (%.4f, %.4f, %.4f), lrate%e, Val.LOSS (%.4f, %.4f, %.4f),\n"
               "    %s, ckpt(%s) abandoned,\n"
               "    EPOCH DURATION: %.2fs\n") % (
            epoch + 1,
            tr_loss_mag, tr_loss_phase, tr_loss, model_lr, val_loss_mag, val_loss_phase, val_loss,
            "NNET Rejected", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      with open(os.path.join(PARAM.SAVE_DIR, PARAM.CHECK_POINT+'_train.log'), 'a+') as f:
        f.writelines(msg+'\n')

      # Start halving when improvement is lower than start_halving_impr
      if (rel_impr_mag < PARAM.start_halving_impr) or (rel_impr_phase < PARAM.start_halving_impr):
        model_lr *= PARAM.halving_factor
        tr_model.assign_lr(sess, model_lr)

      # Stopping criterion
      # if rel_impr_mag < PARAM.end_halving_impr and rel_impr_phase < PARAM.end_halving_impr:
      #   if epoch < PARAM.min_epochs:
      #     tf.logging.info(
      #         "we were supposed to finish, but we continue as "
      #         "min_epochs : %s" % PARAM.min_epochs)
      #     continue
      #   else:
      #     tf.logging.info(
      #         "finished, too small rel. improvement %g,%g" % (rel_impr_mag,rel_impr_phase))
      #     break

    sess.close()
    tf.logging.info("Done training")


def main(argv):
  if not os.path.exists(PARAM.SAVE_DIR):
    os.makedirs(PARAM.SAVE_DIR)
  train()


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
  # mkdir exp && mkdir exp/rnn_speech_enhancement && python3 1_plural_train.py 1 2>&1 | tee exp/rnn_speech_enhancement/nnet_CXXX_train_full.log
  # python3 1_plural_train.py 1 2>&1 | tee exp/rnn_speech_enhancement/nnet_CXXX_train_full.log
