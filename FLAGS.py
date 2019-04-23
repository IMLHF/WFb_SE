from losses import loss
import models.baseline_rnn
import models.recurrent_train_model
import models.threshold_model
import models.threshold_per_frame_model
import models.trainable_logbias_model
import models.training_in_turn_model
import models.individual_bn_model
import models.plural_mask_model
import models.individual_plural_model


class base_config:
  SE_MODEL = models.baseline_rnn.Model_Baseline
  AUDIO_BITS = 16
  NFFT = 256
  FFT_DOT = 129
  INPUT_SIZE = FFT_DOT
  OUTPUT_SIZE = FFT_DOT
  OVERLAP = 128
  FS = 8000
  LEN_WAWE_PAD_TO = FS*3  # Mixed wave length (FS*3 is 3 seconds)
  MEL_BLANCE_COEF = 3.2e7
  MFCC_BLANCE_COEF = 40
  LSTM_num_proj = None
  RNN_SIZE = 512
  MODEL_TYPE = "BLSTM"  # "BLSTM" OR "BGRU"
  LSTM_ACTIVATION = 'tanh'
  MASK_TYPE = "PSM"  # "PSM" or "IRM" or "fixPSM" or "AcutePM" or "PowFixPSM"
  POW_FIX_PSM_COEF = None # for MASK_TYPE = "PowFixPSM"
  PIPLINE_GET_THETA = True
  ReLU_MASK = True
  INPUT_BN = False
  POST_BN =False
  MVN_TYPE = 'BN' # 'BN' or 'BRN'
  SELF_BN = False # if true: batch_normalization(training=True) both when training and decoding

  '''
  LOSS_FUNC_FOR_MAG_SPEC:
    "SPEC_MSE" :
    "MFCC_SPEC_MSE" :
    "MEL_MAG_MSE" :
    "SPEC_MSE_LOWF_EN" :
    "FAIR_SPEC_MSE" :
    "SPEC_MSE_FLEXIBLE_POW_C" :
    "RELATED_MSE" :
    "AUTO_RELATED_MSE" :
      [(y-y_)/(abs(y)+abs(y_))]^2
    "AUTO_RELATED_MSE2" :
      (y-y_)^2/(abs(y)+abs(y_))^1.9
    "AUTO_RELATED_MSE3" :
      [(y-y_)/abs(y)]^2
    "AUTO_RELATED_MSE4" :
      [(y-y_)/(abs(y)+relu(y_))]^2
    "AUTO_RELATED_MSE_USE_COS" :
    "MEL_AUTO_RELATED_MSE" :
  '''
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE"
  '''
  "ABSOLUTE"
  "MAG_WEIGHTED_ABSOLUTE":
  "MIXMAG_WEIGHTED_ABSOLUTE"
  "COS"
  "MAG_WEIGHTED_COS":
  "MIXMAG_WEIGHTED_COS":
  '''
  LOSS_FUNC_FOR_PHASE_SPEC = None
  MAG_LOSS_COEF = None
  PHASE_LOSS_COEF = None
  PI = 3.1415927
  AUTO_RELATED_MSE_AXIS_FIT_DEG = None # for "AUTO_RELATED_MSE"
  COS_AUTO_RELATED_MSE_W = None # for "AUTO_RELATED_MSE_USE_COS"
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_speech_enhancement'
  CHECK_POINT = None

  GPU_RAM_ALLOW_GROWTH = True
  batch_size = 64
  learning_rate = 0.001
  resume_training = 'false'  # set start_epoch = final model ID
  start_epoch = 0
  min_epochs = 15  # Min number of epochs to run trainer without halving.
  max_epochs = 50  # Max number of epochs to run trainer totally.
  halving_factor = 0.5  # Factor for halving.
  max_lr_halving_time = 4
  # Halving when ralative loss is lower than start_halving_impr.
  start_halving_impr = 0.003
  # Stop when relative loss is lower than end_halving_impr.
  end_halving_impr = 0.0005
  # The num of threads to read tfrecords files.
  num_threads_processing_data = 16
  RESTORE_PHASE = 'MIXED'  # 'MIXED','GRIFFIN_LIM',"ESTIMATE".
  GRIFFIN_ITERNUM = 50
  minibatch_size = 200  # batch num to show
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_1_8k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_2_8k'
  NOISE_DIR = '/home/student/work/lhf/alldata/many_noise_8k'
  TFRECORDS_DIR = '/home/student/work/lhf/alldata/irm_data/paper_tfrecords_utt03s_8k_snrmix_wavespan32767_fixAmpBUG'
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  MAX_TFRECORD_FILES_USED = 160  # <=TFRECORDS_NUM
  SHUFFLE = False

  '''
  UTT_SEG_FOR_MIX:(for close condition)
  [260,290] Separate utt to [0:260],[260,290],[290:end]
  to generate train_set, valildation_set and test_cc_set, respectively.
  '''
  UTT_SEG_FOR_MIX = [400, 460]
  DATASET_NAMES = ['train', 'validation', 'test_cc', 'test_oc']
  DATASET_SIZES = [48000, 12000, 6000, 6000]

  INPUT_TYPE = None  # 'mag' or 'logmag'
  LABEL_TYPE = None  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = None  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = None  # should be same to TRAINING_MASK_POSITION
  INIT_MASK_VAL = 0.0
  MIN_LOG_BIAS = 1e-12
  INIT_LOG_BIAS = 0 # real log_bias is MIN_LOG_BIAS+tf.nn.relu(INIT_LOG_BIAS)
  LOG_BIAS_TRAINABLE = False
  #LOG_NORM_MAX = log(LOG_BIAS+MAG_NORM_MAX)
  #LOG_NORM_MIN = log(LOG_BIAS)
  # MAG_NORM_MAX = 70
  MAG_NORM_MAX = 5e5
  # MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP = False

  MIX_METHOD = 'SNR'  # "LINEAR" "SNR"
  MAX_SNR = 5  # 以不同信噪比混合, #并添加10%的纯净语音（尚未添加
  MIN_SNR = -5
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  GET_AUDIO_IN_TEST = False

  TIME_NOSOFTMAX_ATTENTION = False

  # fixed param
  SQUARE_FADE = 'square_fade'
  EXPONENTIAL_FADE = 'exponential_fade'
  EN_EXPONENTIAL_FADE = 'en_exponential_fade'
  THRESHOLD_ON_MASK = 0
  THRESHOLD_ON_SPEC = 1
  ####
  THRESHOLD_FUNC = None # None or 'square_fade' or 'exponential_fade' or 'en_exponential_fade'
  THRESHOLD_POS = None
  INIT_THRESHOLD_RECIPROCAL = 1e-6 # threshold = 1 / INIT_THRESHOLD_RECIPROCAL
  INIT_THRESHOLD_EXP_COEF = None
  THRESHOLD_EXP_TRAINABLE = False

  POW_COEF = None

  USE_CBHG_POST_PROCESSING = False

  SPEC_EST_BIAS = 0.0

class ZSX_TEST(base_config):
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_1_8k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_2_8k'
  NOISE_DIR = '/home/student/work/lhf/alldata/many_noise_8k'
  TFRECORDS_DIR = '/home/student/work/lhf/alldata/irm_data/paper_tfrecords_utt03s_8k_snrmix_wavespan32767_fixAmpBUG'
  CHECK_POINT = 'nnet_ZSX_TEST'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "fixPSM"
  ReLU_MASK = True


class C_RealIRM(base_config): # DONE 15123
  CHECK_POINT = 'nnet_C_RealIRM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"
  PIPLINE_GET_THETA = False
  ReLU_MASK = False


class C_RealPSM(base_config): # DONE 15123
  CHECK_POINT = 'nnet_C_RealPSM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD10(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD10'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 10
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD50(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD50'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 50
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD100(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD500(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD500'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD1000(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD1000'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD1500(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD1500'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1500
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLossAFD2000(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD2000'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 2000
  ReLU_MASK = False


class C_RealIRM_RelativeLossAFD100(base_config): # prepare 15123
  '''
  relative spectrum(mag) MSE AFD100 + IRM
  '''
  CHECK_POINT = 'nnet_C_RealIRM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  MASK_TYPE = "IRM"


class C_ReluIRM(base_config): # DONE 15123
  CHECK_POINT = 'nnet_C_ReluIRM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"
  PIPLINE_GET_THETA = False


class C_ReluPSM(base_config): # DONE 15123
  CHECK_POINT = 'nnet_C_ReluPSM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD50(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD50'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 50
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD100(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD1000(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD1000'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # MASK_TYPE = "PSM" # default


class C_ReluIRM_RelativeLossAFD100(base_config): # prepare 15123
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C_ReluIRM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  MASK_TYPE = "IRM" # default


# improve PSM
# real
# relu


class C_RealPSM_RelativeLoss2AFD100(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE v2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3AFD100(base_config): # prepare 15123
  '''
  relative spectrum(mag) MSE v3
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss4AFD100(base_config): # prepare 15123
  '''
  relative spectrum(mag) MSE v4
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss4AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE4"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealFixPSM_RelativeLoss4AFD100(base_config): # prepare 15123
  '''
  relative spectrum(mag) MSE v4
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss4AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE4"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  MASK_TYPE = 'fixPSM'


PARAM = C_RealPSM_RelativeLoss2AFD100
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)
