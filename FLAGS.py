from losses import loss
import models.baseline_rnn
import models.recurrent_train_model
import models.threshold_model
import models.threshold_per_frame_model
import models.trainable_logbias_model
import models.training_in_turn_model
import models.individual_bn_model


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
  MEL_BLANCE_COEF = 3.2e8
  MFCC_BLANCE_COEF = 40
  LSTM_num_proj = None
  RNN_SIZE = 512
  MODEL_TYPE = "BLSTM"  # "BLSTM" OR "BGRU"
  LSTM_ACTIVATION = 'tanh'
  MASK_TYPE = "PSM"  # "PSM" or "IRM" or "fixPSM" or "AcutePM"
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
    "MEL_SPEC_MSE" :
    "SPEC_MSE_LOWF_EN" :
    "FAIR_SPEC_MSE" :
    "SPEC_MSE_FLEXIBLE_POW_C" :
    "RELATED_MSE" :
    "AUTO_RELATED_MSE" :
    "AUTO_RELATED_MSE_USE_COS" :
  '''
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE"
  '''
  "MAG_WEIGHTED_COS":
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
  start_halving_impr = 0.0003
  resume_training = 'false'  # set start_epoch = final model ID
  start_epoch = 0
  min_epochs = 10  # Min number of epochs to run trainer without halving.
  max_epochs = 15  # Max number of epochs to run trainer totally.
  halving_factor = 0.7  # Factor for halving.
  # Halving when ralative loss is lower than start_halving_impr.
  start_halving_impr = 0.003
  # Stop when relative loss is lower than end_halving_impr.
  end_halving_impr = 0.0005
  # The num of threads to read tfrecords files.
  num_threads_processing_data = 16
  RESTORE_PHASE = 'MIXED'  # 'MIXED','GRIFFIN_LIM'.
  GRIFFIN_ITERNUM = 50
  minibatch_size = 400  # batch num to show
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_1_8k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/home/student/work/lhf/alldata/aishell2_100speaker_list_2_8k'
  NOISE_DIR = '/home/student/work/lhf/alldata/many_noise_8k'
  TFRECORDS_DIR = '/home/student/work/lhf/alldata/irm_data/paper_tfrecords_utt03s_8k_snrmix_wavespan32767'
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


class C001_8_2_full(base_config):
  batch_size = 360
  PROCESS_NUM_GENERATE_TFERCORD = 32
  GENERATE_TFRECORD = False
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/home/root1/worklhf/alldata/full_speaker_aishell2/aishell2_1891speaker_list_1_16k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/home/root1/worklhf/alldata/full_speaker_aishell2/aishell2_100speaker_list_2_16k'
  NOISE_DIR = '/home/root1/worklhf/alldata/many_noise_16k'
  TFRECORDS_DIR = '/home/root1/worklhf/alldata/paper_se/paper_tfrecords_utt03s_16k_fullspeaker_snrmix_wavespan32767'
  DATA_DICT_DIR = '_data/mixed_aishell'
  UTT_SEG_FOR_MIX = [400, 460]
  # DATASET_NAMES = ['train', 'validation', 'test_cc', 'test_oc']
  DATASET_SIZES = [720000, 108000, 6000, 6000]
  FS = 16000
  LEN_WAWE_PAD_TO = FS*3
  NFFT = 512
  FFT_DOT = 257
  INPUT_SIZE = FFT_DOT
  OUTPUT_SIZE = FFT_DOT
  OVERLAP = 256
  CHECK_POINT = 'nnet_C001_8_2_full'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000


class C001_1(base_config): # *DONE 15043
  CHECK_POINT = 'nnet_C001_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"
  PIPLINE_GET_THETA = False


class C001_1_2(base_config): # RUNNING 15123
  CHECK_POINT = 'nnet_C001_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"
  PIPLINE_GET_THETA = False
  ReLU_MASK = False


class C001_2(base_config): # *DONE 15043
  CHECK_POINT = 'nnet_C001_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  # MASK_TYPE = "PSM" # default


class C001_2_2(base_config): # RUNNING 15123
  CHECK_POINT = 'nnet_C001_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C001_2_RT(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_2_RT'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  SE_MODEL = models.recurrent_train_model.Model_Recurrent_Train


class C001_3(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "MFCC_SPEC_MSE"
  SPEC_LOSS_COEF = 0.5
  MFCC_LOSS_COEF = 0.5


class C001_3_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_3_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "MFCC_SPEC_MSE"
  SPEC_LOSS_COEF = 0.8
  MFCC_LOSS_COEF = 0.2


class C001_4(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.5
  MEL_LOSS_COEF = 0.5


class C001_4_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.8
  MEL_LOSS_COEF = 0.2


class C001_4_3(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.2
  MEL_LOSS_COEF = 0.8


class C001_5(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_LOWF_EN"
  # MASK_TYPE = "PSM" # default


class C001_6(base_config): # DONE 15041
  '''
  fair spectrum(mag or log) MSE
  '''
  CHECK_POINT = 'nnet_C001_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "FAIR_SPEC_MSE"
  # MASK_TYPE = "PSM" # default


class C001_6_2(base_config): # DONE 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_FLEXIBLE_POW_C"
  POW_COEF = 1
  # MASK_TYPE = "PSM" # default


class C001_6_3(base_config): # DONE 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_FLEXIBLE_POW_C"
  POW_COEF = 1.5


class C001_6_4(base_config): # DONE 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_FLEXIBLE_POW_C"
  POW_COEF = 2.5


class C001_6_5(base_config): # DONE 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_FLEXIBLE_POW_C"
  POW_COEF = 3


class C001_7(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_7'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  RELATED_MSE_IGNORE_TH = 1e-12
  # MASK_TYPE = "PSM" # default


class C001_7_2(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_7_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  RELATED_MSE_IGNORE_TH = 1e-6
  # MASK_TYPE = "PSM" # default


class C001_7_3(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_7_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  RELATED_MSE_IGNORE_TH = 1e-4
  # MASK_TYPE = "PSM" # default


class C001_8_1(base_config): # *DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  # MASK_TYPE = "PSM" # default


class C001_8_2(base_config): # *DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # MASK_TYPE = "PSM" # default


class C001_8_2_fixPSM(base_config): # *DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_fixPSM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = "fixPSM"


class C001_8_2_realFixPSM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_realFixPSM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = "fixPSM"
  ReLU_MASK = False


class C001_8_2_realPSM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_realPSM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  ReLU_MASK = False


class C001_8_2_reluIRM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_reluIRM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = 'IRM'
  PIPLINE_GET_THETA = False


class C001_8_2_realIRM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_realIRM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = 'IRM'
  ReLU_MASK = False
  PIPLINE_GET_THETA = False


class C001_8_2_reluAcutePM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_reluAcutePM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = "AcutePM" # default
  ReLU_MASK = True


class C001_8_2_realAcutePM(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_realAcutePM'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MASK_TYPE = "AcutePM" # default
  ReLU_MASK = False


class C001_8_2_fourLayerRNN(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  RNN_LAYER = 4
  CHECK_POINT = 'nnet_C001_8_2_fourLayerRNN'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # MASK_TYPE = "PSM" # default


class C001_8_2_2(base_config):  # DONE 15043
  CHECK_POINT = 'nnet_C001_8_2_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # INIT_MASK_VAL = 1.0
  USE_CBHG_POST_PROCESSING = True
  DOUBLE_LOSS = False
  # learning_rate = 0.0001


class C001_8_2_3(base_config):  # DONE 15123
  CHECK_POINT = 'nnet_C001_8_2_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # INIT_MASK_VAL = 1.0
  USE_CBHG_POST_PROCESSING = True
  DOUBLE_LOSS = True
  CBHG_LOSS_COEF1 = 0.5
  CBHG_LOSS_COEF2 = 0.5
  # learning_rate = 0.0001


class C001_8_2_4(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE GRU
  '''
  CHECK_POINT = 'nnet_C001_8_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MODEL_TYPE = "BGRU"
  # MASK_TYPE = "PSM" # default


class C001_8_2_5(base_config): # DONE 15043
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_2_5'
  batch_size = 1
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # MASK_TYPE = "PSM" # default


class C001_8_3(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 10000
  # MASK_TYPE = "PSM" # default


class C001_8_4(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 2000
  # MASK_TYPE = "PSM" # default


class C001_8_5(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 5000
  # MASK_TYPE = "PSM" # default


class C001_8_6(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 7000
  # MASK_TYPE = "PSM" # default


class C001_8_7(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE
  '''
  CHECK_POINT = 'nnet_C001_8_7'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  # MASK_TYPE = "PSM" # default


class C001_8_8(base_config): # *DONE 15123
  '''
  relative spectrum(mag) MSE with INPUT_BN
  '''
  CHECK_POINT = 'nnet_C001_8_8'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  INPUT_BN = True
  SELF_BN = True


class C001_8_9(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE with INPUT_BN
  '''
  CHECK_POINT = 'nnet_C001_8_9'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  INPUT_BN = True
  SELF_BN = True


class C001_8_10(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE with INPUT_BReN
  '''
  CHECK_POINT = 'nnet_C001_8_10'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  INPUT_BN = True
  MVN_TYPE = 'BRN' # 'BN' or 'BRN'
  SELF_BN = True


class C001_8_11(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE with INDIVIDUAL_BN
  '''
  SE_MODEL=models.individual_bn_model.INDIVIDUAL_BN_MODEL
  CHECK_POINT = 'nnet_C001_8_11'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  USE_ESTIMATED_MEAN_VAR = False
  BN_KEEP_DIMS=[-2,-1]


class C001_8_11_2(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE with INDIVIDUAL_BN
  '''
  SE_MODEL=models.individual_bn_model.INDIVIDUAL_BN_MODEL
  CHECK_POINT = 'nnet_C001_8_11_2'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  USE_ESTIMATED_MEAN_VAR = False
  BN_KEEP_DIMS=[-1]


class C001_8_12(base_config): # CKPT Same To C001_8_11
  '''
  relative spectrum(mag) MSE with INDIVIDUAL_BN
  '''
  SE_MODEL=models.individual_bn_model.INDIVIDUAL_BN_MODEL
  CHECK_POINT = 'nnet_C001_8_12'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  USE_ESTIMATED_MEAN_VAR = True
  BN_KEEP_DIMS=[-1,-2]


class C001_8_12_2(base_config): # CKPT Same To C001_8_11_2
  '''
  relative spectrum(mag) MSE with INDIVIDUAL_BN
  '''
  SE_MODEL=models.individual_bn_model.INDIVIDUAL_BN_MODEL
  CHECK_POINT = 'nnet_C001_8_12_2'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  USE_ESTIMATED_MEAN_VAR = True
  BN_KEEP_DIMS=[-1]


class C001_8_13(base_config): # DONE 15041
  '''
  relative spectrum(mag) MSE with POST_BN
  '''
  CHECK_POINT = 'nnet_C001_8_13'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  POST_BN = True
  SELF_BN = False


class C001_9_1(base_config): # DONE 15123
  '''
  cos relative spectrum(mag) MSE
  '''
  PI = 3.1415927
  CHECK_POINT = 'nnet_C001_9_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE_USE_COS"
  COS_AUTO_RELATED_MSE_W = PI
  # MASK_TYPE = "PSM" # default


class C001_9_2(base_config): # DONE 15123
  '''
  cos relative spectrum(mag) MSE
  '''
  PI = 3.1415927
  CHECK_POINT = 'nnet_C001_9_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE_USE_COS"
  COS_AUTO_RELATED_MSE_W = PI*1.25


class C002_1(base_config): # DONE 15043
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.SQUARE_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 50


class C002_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 50


class C002_2_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_2_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 1e6
  max_epochs = 15


class C002_2_3(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_2_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 0.0
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  max_epochs = 15


class C002_3_1(base_config): # DONE 15043
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_3_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = False
  INIT_THRESHOLD_EXP_COEF = 2
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 10
  max_epochs = 50


class C002_3_2(base_config): # DONE 15043
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_3_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = False
  INIT_THRESHOLD_EXP_COEF = 4
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 10
  max_epochs = 50


class C002_4(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.SQUARE_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_MASK
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 50


class C002_5(base_config): # diverge DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_MASK
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 50


class C002_6(base_config): # diverge DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = False
  INIT_THRESHOLD_EXP_COEF = 2
  THRESHOLD_POS = base_config.THRESHOLD_ON_MASK
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 50


class C002_5_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_5_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_MASK
  INIT_THRESHOLD_RECIPROCAL = 10
  max_epochs = 50


class C002_6_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_6_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = False
  INIT_THRESHOLD_EXP_COEF = 2
  THRESHOLD_POS = base_config.THRESHOLD_ON_MASK
  INIT_THRESHOLD_RECIPROCAL = 10
  max_epochs = 50


class C002_7_1(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_7_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = True
  INIT_THRESHOLD_EXP_COEF = 1
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 15


class C002_7_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_model.Threshold_Model
  CHECK_POINT = 'nnet_C002_7_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EN_EXPONENTIAL_FADE
  THRESHOLD_EXP_TRAINABLE = True
  INIT_THRESHOLD_EXP_COEF = 1
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 1e6
  max_epochs = 15


class C002_8_1(base_config): # DONE 15041
  SE_MODEL = models.threshold_per_frame_model.Frame_Threshold_Model
  CHECK_POINT = 'nnet_C002_8_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 0
  max_epochs = 15


class C002_8_2(base_config): # DONE 15041
  SE_MODEL = models.threshold_per_frame_model.Frame_Threshold_Model
  CHECK_POINT = 'nnet_C002_8_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 1e6
  max_epochs = 15


class C003_1(base_config): # DONE 15041
  '''
  time-nosoftmax-attention
  '''
  CHECK_POINT = 'nnet_C003_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  TIME_NOSOFTMAX_ATTENTION = True


class C004_1(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_1'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0


class C004_1_2(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_2'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e-2


class C004_1_3(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_3'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e2


class C004_1_4(base_config):  # **DONE 15041
  CHECK_POINT = 'nnet_C004_1_4'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4


class C004_1_5(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_5'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e5


class C004_1_6(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_6'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e6


class C004_1_7(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_7'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4-5e3 # 5e3


class C004_1_8(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_8'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4+5e3


class C004_1_4_2(base_config):  # DONE 15041
  CHECK_POINT = 'nnet_C004_1_4_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4


class C004_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e-2


class C004_2_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_2'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1


class C004_2_3(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e2


class C004_2_4(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4


class C004_2_5(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e5


class C004_2_7(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_7'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4-5e3


class C004_2_8(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C004_2_8'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4+5e3


class C005_1(base_config): # DONE 15043
  CHECK_POINT = 'nnet_C005_1'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4
  LOSS_FUNC_FOR_MAG_SPEC = "SPEC_MSE_FLEXIBLE_POW_C"
  POW_COEF = 1


class C006_1_1(base_config): # DONE 15041
  SE_MODEL = models.trainable_logbias_model.Trainable_Logbias_Model
  CHECK_POINT = 'nnet_C006_1_1'
  INPUT_TYPE = 'mag'
  LABEL_TYPE = 'logmag'
  TRAINING_MASK_POSITION = 'mag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  # LOSS_FUNC_FOR_MAG_SPEC = "LOG_BIAS_DE"
  INIT_LOG_BIAS = 1e4


class C006_1_2(base_config): # DONE 15041
  SE_MODEL = models.trainable_logbias_model.Trainable_Logbias_Model
  CHECK_POINT = 'nnet_C006_1_1delogbias'
  INPUT_TYPE = 'mag'
  LABEL_TYPE = 'logmag'
  TRAINING_MASK_POSITION = 'mag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "LOG_BIAS_DE"
  INIT_LOG_BIAS = 1e4


class C006_2_1(base_config): # DONE 15043
  '''
  step1 : train MASKNET
  step2 : train LOGBIASNET
  '''
  RNN_SIZE = None
  RNN_LAYER = None
  LSTM_num_proj = None
  LSTM_ACTIVATION = None
  SE_MODEL = models.training_in_turn_model.ALTER_Training_Model
  CHECK_POINT = 'nnet_C006_2_1'
  learning_rate_logbiasnet = 0.001
  learning_rate_masknet = 0.001
  RNN_SIZE_LOGBIAS = 256
  RNN_SIZE_MASK = 512
  RNN_LAYER_LOGBIAS = 2
  RNN_LAYER_MASK = 2
  LSTM_num_proj_LOGBIAS = None
  LSTM_num_proj_MASK = None
  LSTM_ACTIVATION_LOGBIAS = 'tanh'
  LSTM_ACTIVATION_MASK = 'tanh'
  INPUT_TYPE = 'mag'
  TRAINING_MASK_POSITION = 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4
  TRAIN_TYPE = 'MASKNET' # 'LOGBIASNET' 'MASKNET' 'BOTH'


class C007_1(base_config): # prepare 15041
  CHECK_POINT = 'nnet_C007_1'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  LOSS_FUNC_FOR_PHASE_SPEC = "COS" # ****
  MAG_LOSS_COEF = 1.0
  PHASE_LOSS_COEF = 1.0
  PHASE_LOSS_INDEX = 2.0 # ****
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  FFT_DOT = 129
  INPUT_SIZE = FFT_DOT*2
  OUTPUT_SIZE = FFT_DOT*2
  RNN_SIZE = 512 # ****
  # MASK_TYPE = "PSM" # default

class C007_2(base_config): # prepare 15041
  CHECK_POINT = 'nnet_C007_2'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  LOSS_FUNC_FOR_PHASE_SPEC = "MAG_WEIGHTED_COS"
  MAG_LOSS_COEF = 1.0
  PHASE_LOSS_COEF = 1.0
  PHASE_LOSS_INDEX = 2.0
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  FFT_DOT = 129
  INPUT_SIZE = FFT_DOT*2
  OUTPUT_SIZE = FFT_DOT*2
  RNN_SIZE = 512
  # MASK_TYPE = "PSM" # default


class C007_3(base_config): # prepare 15041
  CHECK_POINT = 'nnet_C007_3'
  # INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  # LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  # TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  # DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  LOSS_FUNC_FOR_PHASE_SPEC = "MAG_WEIGHTED_COS or COS"
  MAG_LOSS_COEF = 1.0
  PHASE_LOSS_COEF = 1.0
  PHASE_LOSS_INDEX = 2.0
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  FFT_DOT = 129
  INPUT_SIZE = FFT_DOT*2
  OUTPUT_SIZE = FFT_DOT*2
  RNN_SIZE = 1024
  # MASK_TYPE = "PSM" # default

PARAM = C007_1
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)
