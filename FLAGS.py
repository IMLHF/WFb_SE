from losses import loss
import models.baseline_rnn


class base_config:
  SE_MODEL = models.baseline_rnn.Model_Baseline
  AUDIO_BITS = 16
  NFFT = 256
  OVERLAP = 128
  FS = 8000
  INPUT_SIZE = 129
  OUTPUT_SIZE = 129
  LSTM_num_proj = None
  RNN_SIZE = 512
  MODEL_TYPE = "BLSTM"  # "BLSTM" OR "BGRU"
  LSTM_ACTIVATION = 'tanh'
  MASK_TYPE = "PSM"  # "PSM" or "IRM"
  LOSS_FUNC = "SPEC_MSE" # "SPEC_MSE" or "MFCC_SPEC_MSE"
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
  CLOSE_CONDATION_SPEAKER_LIST_DIR = '/home/room/work/lhf/alldata/aishell2_100speaker_list_1_8k'
  OPEN_CONDATION_SPEAKER_LIST_DIR = '/home/room/work/lhf/alldata/aishell2_100speaker_list_2_8k'
  # NOISE_DIR = '/all_data/many_noise' # for docker
  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise_8k'
  DATA_DICT_DIR = '_data/mixed_aishell'
  # TFRECORDS_DIR = '/all_data/feature_tfrecords' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/paper_tfrecords_utt03s_8k_snrmix_wavespan32767'
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
  LOG_BIAS_TRAINABEL = False
  #LOG_NORM_MAX = log(LOG_BIAS+MAG_NORM_MAX)
  #LOG_NORM_MIN = log(LOG_BIAS)
  # MAG_NORM_MAX = 70
  MAG_NORM_MAX = 5e5
  # MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP = False

  MIX_METHOD = 'SNR'  # "LINEAR" "SNR"
  MAX_SNR = 5  # 以不同信噪比混合, (是否添加纯净语音到训练集)
  MIN_SNR = -5
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

  GET_AUDIO_IN_TEST = False


class C_ReluChangeToWeightedSoftmax(base_config):
  CHECK_POINT = 'nnet_C_ReluChangeToWeightedSoftmax'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION

class C001(base_config): #
  CHECK_POINT = 'nnet_C001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  MASK_TYPE = "IRM"
  '''
  iter4 PESQ: 0.4
  '''

class C001_2(base_config): #
  CHECK_POINT = 'nnet_C001_2'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  # MASK_TYPE = "PSM" # default
  '''
  iter4 PESQ: 0.4
  '''

class C001_3(base_config): #
  CHECK_POINT = 'nnet_C001_3'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  LOSS_FUNC = "MFCC_SPEC_MSE"


class C002(base_config):  #
  CHECK_POINT = 'nnet_C002'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0


class C003(base_config):
  CHECK_POINT = 'nnet_C003'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0

# 002和003对比看mask最佳位置，001和003对比看对数谱和幅度谱哪个做loss较好

class C004(base_config):  #
  CHECK_POINT = 'nnet_C004'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 50

class C005(base_config):
  CHECK_POINT = 'nnet_C005'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 50

class C006(base_config):
  CHECK_POINT = 'nnet_C006'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 100

class C007(base_config):
  CHECK_POINT = 'nnet_C006'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 100

class C008(base_config):
  CHECK_POINT = 'nnet_C006'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1000

class C009(base_config):
  CHECK_POINT = 'nnet_C006'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 1000

class CXX(base_config):  #
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0
  LOG_BIAS_TRAINABEL = True


PARAM = C001
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)
