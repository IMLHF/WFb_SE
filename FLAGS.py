from losses import loss
import models.baseline_rnn
import models.recurrent_train_model
import models.threshold_model
import models.threshold_per_frame_model


class base_config:
  SE_MODEL = models.baseline_rnn.Model_Baseline
  AUDIO_BITS = 16
  NFFT = 256
  OVERLAP = 128
  FS = 8000
  MEL_BLANCE_COEF = 3.2e8
  MFCC_BLANCE_COEF = 40
  INPUT_SIZE = 129
  OUTPUT_SIZE = 129
  LSTM_num_proj = None
  RNN_SIZE = 512
  MODEL_TYPE = "BLSTM"  # "BLSTM" OR "BGRU"
  LSTM_ACTIVATION = 'tanh'
  MASK_TYPE = "PSM"  # "PSM" or "IRM"
  '''
  LOSS_FUNC:
    "SPEC_MSE" :
    "MFCC_SPEC_MSE" :
    "MEL_SPEC_MSE" :
    "SPEC_MSE_LOWF_EN" :
    "FAIR_SPEC_MSE" :
    "SPEC_MSE_EMPHASIZE_LOWER_V" :
  '''
  LOSS_FUNC = "SPEC_MSE"
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
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','GRIFFIN_LIM'.
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

  LEN_WAWE_PAD_TO = FS*3  # Mixed wave length (FS*3 is 3 seconds)
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
  DEFAULT_LOG_BIAS = 1e-12
  INIT_LOG_BIAS = 0 # real log_bias is DEFAULT_LOG_BIAS+tf.nn.relu(INIT_LOG_BIAS)
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


class C001_1(base_config): # *DONE 15041
  CHECK_POINT = 'nnet_C001_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"


class C001_2(base_config): # *DONE 15041
  CHECK_POINT = 'nnet_C001_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
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
  LOSS_FUNC = "MFCC_SPEC_MSE"
  SPEC_LOSS_COEF = 0.5
  MFCC_LOSS_COEF = 0.5


class C001_3_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_3_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "MFCC_SPEC_MSE"
  SPEC_LOSS_COEF = 0.8
  MFCC_LOSS_COEF = 0.2


class C001_4(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.5
  MEL_LOSS_COEF = 0.5


class C001_4_2(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.8
  MEL_LOSS_COEF = 0.2


class C001_4_3(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_4_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "MEL_SPEC_MSE"
  SPEC_LOSS_COEF = 0.2
  MEL_LOSS_COEF = 0.8


class C001_5(base_config): # DONE 15041
  CHECK_POINT = 'nnet_C001_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_LOWF_EN"
  # MASK_TYPE = "PSM" # default


class C001_6(base_config): # *DONE 15041
  '''
  fair spectrum(mag or log) MSE
  '''
  CHECK_POINT = 'nnet_C001_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "FAIR_SPEC_MSE"
  # MASK_TYPE = "PSM" # default


class C001_6_2(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1
  # MASK_TYPE = "PSM" # default


class C001_6_3(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1/1.5


class C001_6_4(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1/2


class C001_6_5(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1/3


class C001_6_6(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1/5


class C001_6_7(base_config): # prepare 15041
  '''
  MSE emphasize lower value
  '''
  CHECK_POINT = 'nnet_C001_6_7'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "SPEC_MSE_EMPHASIZE_LOWER_V"
  POW_COEF = 1/9


class C002_1(base_config): # DONE 15043
  CLOSE_CONDATION_SPEAKER_LIST_DIR = base_config.CLOSE_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  OPEN_CONDATION_SPEAKER_LIST_DIR = base_config.OPEN_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  NOISE_DIR = base_config.NOISE_DIR.replace('room','student')
  TFRECORDS_DIR = base_config.TFRECORDS_DIR.replace('room','student')
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


class C002_2(base_config): # **DONE 15041
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
  CHECK_POINT = 'nnet_C002_2_1'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  THRESHOLD_FUNC = base_config.EXPONENTIAL_FADE
  THRESHOLD_POS = base_config.THRESHOLD_ON_SPEC
  INIT_THRESHOLD_RECIPROCAL = 1e6
  max_epochs = 15


class C002_3_1(base_config): # DONE 15043
  CLOSE_CONDATION_SPEAKER_LIST_DIR = base_config.CLOSE_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  OPEN_CONDATION_SPEAKER_LIST_DIR = base_config.OPEN_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  NOISE_DIR = base_config.NOISE_DIR.replace('room','student')
  TFRECORDS_DIR = base_config.TFRECORDS_DIR.replace('room','student')
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
  CLOSE_CONDATION_SPEAKER_LIST_DIR = base_config.CLOSE_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  OPEN_CONDATION_SPEAKER_LIST_DIR = base_config.OPEN_CONDATION_SPEAKER_LIST_DIR.replace('room','student')
  NOISE_DIR = base_config.NOISE_DIR.replace('room','student')
  TFRECORDS_DIR = base_config.TFRECORDS_DIR.replace('room','student')
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


class C002_9_1(base_config): # prepare 15041
  SE_MODEL = models.threshold_per_frame_model.Frame_Threshold_Model
  CHECK_POINT = 'nnet_C002_9_1'
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


class C002_9_2(base_config): # prepare 15041
  SE_MODEL = models.threshold_per_frame_model.Frame_Threshold_Model
  CHECK_POINT = 'nnet_C002_9_2'
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


class C004_1_3(base_config):  # *DONE 15041
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


class C004_1_5(base_config):  # RUNNING 15041
  CHECK_POINT = 'nnet_C004_1_5'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e5


class C004_1_6(base_config):  # RUNNING 15041
  CHECK_POINT = 'nnet_C004_1_6'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e6


class C004_1_4_2(base_config):  # *RUNNING 15041
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


class C004_2_3(base_config): # RUNNING 15041
  CHECK_POINT = 'nnet_C004_2_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e2


class C004_2_4(base_config): # RUNNING 15041
  CHECK_POINT = 'nnet_C004_2_4'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e4


class C004_2_5(base_config): # prepare 15041
  CHECK_POINT = 'nnet_C004_2_5'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e5


class C004_2_6(base_config): # prepare 15041
  CHECK_POINT = 'nnet_C004_2_6'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1e6

PARAM = C004_2_4
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)
