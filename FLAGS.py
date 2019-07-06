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
      [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
    "AUTO_RELATED_MSE2" :
      [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
    "AUTO_RELATED_MSE3" :
      [|y-y_|/((A*|y|)^C1+B)]^C2
    "AUTO_RELATED_MSE4" :
      [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+relu(sign(y)*y_))]^2
    "AUTO_RELATED_MSE5" :
      [relu(y-y_)/(2|y|+relu(y-y_))]^2+[relu(y-y_)/(|y|+relu(|y|-y+y_))]^2
    "AUTO_RELATED_MSE6" :
      [(y-y_)/(A*sqrt(|y|*|y_|)^C1+B)]^C2
    "AUTO_RELATED_MSE7" :
      [(y-y_)/((A1*sqrt((|y|+|y_|)*|y_|)+A2*(|y|+|y_|)^C1+B)]^C2
    "AUTO_RELATED_MSE8" :
      [|y-y_|/((A*(|y|+|y_|)^C1+B)]^C2
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
  AUTO_RELATIVE_LOSS3_A = None # for "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_B = None
  AUTO_RELATIVE_LOSS3_C1 = None
  AUTO_RELATIVE_LOSS3_C2 = None
  AUTO_RELATIVE_LOSS6_A = None
  AUTO_RELATIVE_LOSS6_B = None
  AUTO_RELATIVE_LOSS6_C1 = None
  AUTO_RELATIVE_LOSS6_C2 = None
  AUTO_RELATIVE_LOSS7_A1 = None
  AUTO_RELATIVE_LOSS7_A2 = None
  AUTO_RELATIVE_LOSS7_B = None
  AUTO_RELATIVE_LOSS7_C1 = None
  AUTO_RELATIVE_LOSS7_C2 = None
  AUTO_RELATIVE_LOSS8_A = None
  AUTO_RELATIVE_LOSS8_B = None
  AUTO_RELATIVE_LOSS8_C1 = None
  AUTO_RELATIVE_LOSS8_C2 = None
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
  end_halving_impr = 0.00000
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

class C001_8_2_full_adddevnoise_enac(base_config):
  resume_training = 'true'
  start_epoch = 25
  min_epochs = 26
  max_epochs = 26
  batch_size = 360
  PROCESS_NUM_GENERATE_TFERCORD = 16
  GENERATE_TFRECORD = False
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/speakerlist'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/speakerlist'
  NOISE_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/noise'
  TFRECORDS_DIR = '/data/datalhf/irm_data/accept_tfrecord'
  DATA_DICT_DIR = '_data/mixed_aishell'
  UTT_SEG_FOR_MIX = [2, 4]
  # DATASET_NAMES = ['train', 'validation', 'test_cc', 'test_oc']
  DATASET_SIZES = [360000, 720, 600, 600]
  FS = 16000
  LEN_WAWE_PAD_TO = FS*3
  NFFT = 512
  FFT_DOT = 257
  INPUT_SIZE = FFT_DOT
  OUTPUT_SIZE = FFT_DOT
  OVERLAP = 256
  CHECK_POINT = 'nnet_C001_8_2_full_adddevnoise_enac'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MAX_TFRECORD_FILES_USED = 16
  learning_rate = 0.0001

class C001_8_2_full_adddevnoise(base_config):
  # ls awk -F ' ' '{print $1,$1}' || ls awk -F ' ' '{print $1" "$1}'
  batch_size = 360
  PROCESS_NUM_GENERATE_TFERCORD = 16
  GENERATE_TFRECORD = False
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/data/datalhf/full_speaker_aishell2/aishell2_1891speaker_list_1_16k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/data/datalhf/full_speaker_aishell2/aishell2_100speaker_list_2_16k'
  NOISE_DIR = '/data/datalhf/many_noise_16k_adddev'
  TFRECORDS_DIR = '/data/datalhf/irm_data/tfrecords_utt03s_16k_fullspeaker_snrmix_wavespan32767_adddevnoise'
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
  CHECK_POINT = 'nnet_C001_8_2_full_adddevnoise'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000

class C001_8_2_full_en_ac(base_config):
  resume_training = 'true'
  start_epoch = 15
  min_epochs = 16
  max_epochs = 16
  batch_size = 360
  PROCESS_NUM_GENERATE_TFERCORD = 16
  GENERATE_TFRECORD = False
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/speakerlist'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/speakerlist'
  NOISE_DIR = '/fast/worklhf/lhf/alldata/accept_data_16k/noise'
  TFRECORDS_DIR = '/data/datalhf/irm_data/accept_tfrecord'
  DATA_DICT_DIR = '_data/mixed_aishell'
  UTT_SEG_FOR_MIX = [2, 4]
  # DATASET_NAMES = ['train', 'validation', 'test_cc', 'test_oc']
  DATASET_SIZES = [360000, 720, 600, 600]
  FS = 16000
  LEN_WAWE_PAD_TO = FS*3
  NFFT = 512
  FFT_DOT = 257
  INPUT_SIZE = FFT_DOT
  OUTPUT_SIZE = FFT_DOT
  OVERLAP = 256
  CHECK_POINT = 'nnet_C001_8_2_full_en_ac'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  MAX_TFRECORD_FILES_USED = 16
  learning_rate = 0.0001

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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLossAFD2000'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 2000
  ReLU_MASK = False


class C_RealIRM_RelativeLossAFD100(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE AFD100 + IRM
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
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


class C_RealIRM_RelativeLossAFD500(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE AFD500 + IRM
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealIRM_RelativeLossAFD500'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
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


class C_ReluIRM_NOAFD(base_config): # RUNNING 15043
  CHECK_POINT = 'nnet_C_ReluIRM_NOAFD'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  MASK_TYPE = "IRM"
  RELATED_MSE_IGNORE_TH = 0.0

class C_ReluPSM_NOAFD(base_config): # RUNNING 15043
  CHECK_POINT = 'nnet_C_ReluPSM_NOAFD'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  RELATED_MSE_IGNORE_TH = 0.0
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD50(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  # GET_AUDIO_IN_TEST = True
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD50'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 50
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD100(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD500(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD500'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD1000(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD1000'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD500_rl4(base_config): # RUNNING 15041
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD500_rl4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  RNN_LAYER = 4
  # MASK_TYPE = "PSM" # default


class C_ReluPSM_RelativeLossAFD500_rs1024(base_config): # RUNNING 15041
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD500_rl4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  RNN_SIZE = 1024


class C_ReluPSM_RelativeLossAFD500_rl6(base_config): # RUNNING 15041
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluPSM_RelativeLossAFD500_rl6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  RNN_LAYER = 6
  # MASK_TYPE = "PSM" # default


class C_ReluIRM_RelativeLossAFD100(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_ReluIRM_RelativeLossAFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  MASK_TYPE = "IRM" # default


class C_ReluIRM_RelativeLossAFD500(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  # GET_AUDIO_IN_TEST = True
  CHECK_POINT = 'nnet_C_ReluIRM_RelativeLossAFD500'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  MASK_TYPE = "IRM" # default


# improve PSM
# real
# relu


class C_RealPSM_RelativeLoss2AFD100(base_config): # *DONE 15123
  '''
  relative spectrum(mag) MSE v2
  [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  LINEAR_BROKER = 1.05
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss2AFD100_LB1_2(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v2
  [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD100_LB1_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  LINEAR_BROKER = 1.2
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss2AFD1000_LB1_2(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v2
  [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD1000_LB1_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  LINEAR_BROKER = 1.2
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss2AFD2000_LB1_2(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v2
  [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD2000_LB1_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 2000
  LINEAR_BROKER = 1.2
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss2AFD1000_LB1_1(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v2
  [|y-y_|^lb/(1/AFD+(1-1/AFD)*(|y|+|y_|)]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss2AFD1000_LB1_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE2"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  LINEAR_BROKER = 1.1
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_001(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.1 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_002(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_002'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_003(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_003'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.15 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_004(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_004'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 0.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_005(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_005'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_006(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_006'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.02 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_007(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_007'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.07 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_008(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_008'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.7 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_009(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_009'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.1 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_010(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_010'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 0.9 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_011(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_011'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.1 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.1 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_012(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_012'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 2.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.1 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.2 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 1.8 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_013(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_013'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 1.8 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss3_014(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_014'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.05 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.2 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False


class C_RealPSM_RelativeLoss3_015(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_015'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 2.0 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.1 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.2 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False


class C_RealPSM_RelativeLoss3_016(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v3
  [|y-y_|/((A*|y|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss3_016'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE3"
  AUTO_RELATIVE_LOSS3_A = 1.5 # A: [1左右] 控制关注程度的变化速度，越大损失函数的开口变化速度越快
  AUTO_RELATIVE_LOSS3_B = 0.01 # B: [0.001-0.2] 控制|y|趋向于无穷小时的开口大小,B越小开口越小;开口太小不收敛,开口太大效果差(小值的关注度不够)
  AUTO_RELATIVE_LOSS3_C1 = 1.0 # C1: 为1时, 底部等高线为直线; 小于1时向外弯曲, 类似x^2; 大于1时向内弯曲.
  AUTO_RELATIVE_LOSS3_C2 = 2.0 # C2: 次幂,越大容错率越大,容易收敛,收敛后效果较差;越小容错率越小,不以收敛,收敛后效果较好.
  ReLU_MASK = False


# 初始的ReLoss_MSE，对应的A大一些，B也大一些，C1=1.0

class C_RealPSM_RelativeLoss4AFD100(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE v4
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+relu(sign(y)*y_))]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss4AFD100_001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE4"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 100
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss4AFD500(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE v4
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+relu(sign(y)*y_))]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss4AFD100_002'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE4"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 500
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss4AFD1000(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE v4
  [(y-y_)/(1/AFD+(1-1/AFD)*(|y|+relu(sign(y)*y_))]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss4AFD100_003'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE4"
  AUTO_RELATED_MSE_AXIS_FIT_DEG = 1000
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss5(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v5
  [relu(y-y_)/(2|y|+relu(y-y_))]^2+[relu(y-y_)/(|y|+relu(|y|-y+y_))]^2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE5"
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss6_001(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v6
  # [(y-y_)/(sqrt(|y|*|y_|)+min_refer)]^2
  [(y-y_)/(A*sqrt(|y|*|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss6_001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE6"
  AUTO_RELATIVE_LOSS6_A = 1.0
  AUTO_RELATIVE_LOSS6_B = 0.05
  AUTO_RELATIVE_LOSS6_C1 = 1.0
  AUTO_RELATIVE_LOSS6_C2 = 2.0
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss6_002(base_config): # diverge 15123
  '''
  relative spectrum(mag) MSE v6
  [(y-y_)/(A*sqrt(|y|*|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss6_002'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE6"
  AUTO_RELATIVE_LOSS6_A = 1.0
  AUTO_RELATIVE_LOSS6_B = 0.01
  AUTO_RELATIVE_LOSS6_C1 = 1.0
  AUTO_RELATIVE_LOSS6_C2 = 2.0
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss7_AFD100(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v7
  [(y-y_)/((A1*sqrt((|y|+|y_|)*|y_|)+A2*(|y|+|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss7_AFD100'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE7"
  AUTO_RELATIVE_LOSS7_A1 = 0.5
  AUTO_RELATIVE_LOSS7_A2 = 0.5
  AUTO_RELATIVE_LOSS7_B = 0.01
  AUTO_RELATIVE_LOSS7_C1 = 1.0
  AUTO_RELATIVE_LOSS7_C2 = 2.0
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss8_001(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v8 (transition of v1)
  [|y-y_|/(A*(|y|+|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss8_001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE8"
  AUTO_RELATIVE_LOSS8_A = 1.5
  AUTO_RELATIVE_LOSS8_B = 0.01
  AUTO_RELATIVE_LOSS8_C1 = 1.0
  AUTO_RELATIVE_LOSS8_C2 = 2.0
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss8_002(base_config): # DONE 15123
  '''
  relative spectrum(mag) MSE v8 (transition of v1)
  [|y-y_|/(A*(|y|+|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss8_002'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE8"
  AUTO_RELATIVE_LOSS8_A = 0.5
  AUTO_RELATIVE_LOSS8_B = 0.01
  AUTO_RELATIVE_LOSS8_C1 = 1.0
  AUTO_RELATIVE_LOSS8_C2 = 2.0
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C_RealPSM_RelativeLoss8_003(base_config): # RUNNING 15123
  '''
  relative spectrum(mag) MSE v8 (transition of v1)
  [|y-y_|/(A*(|y|+|y_|)^C1+B)]^C2
  '''
  CHECK_POINT = 'nnet_C_RealPSM_RelativeLoss8_003'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "AUTO_RELATED_MSE8"
  AUTO_RELATIVE_LOSS8_A = 1.0
  AUTO_RELATIVE_LOSS8_B = 0.01
  AUTO_RELATIVE_LOSS8_C1 = 1.0
  AUTO_RELATIVE_LOSS8_C2 = 1.5
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


class C001_7_3_retest(base_config): # DONE 15043
  CHECK_POINT = 'nnet_CC001_7_3_retest'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC_FOR_MAG_SPEC = "RELATED_MSE"
  RELATED_MSE_IGNORE_TH = 0.0001
  ReLU_MASK = False
  # MASK_TYPE = "PSM" # default


PARAM = C001_8_2_full_adddevnoise_enac
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)
