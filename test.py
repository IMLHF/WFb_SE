from utils import audio_tool,spectrum_tool
import tensorflow as tf
# from models.baseline_rnn import rm_norm_mag_spec, norm_mag_spec, rm_norm_logmag_spec, norm_logmag_spec, normedLogmag2normedMag,normedMag2normedLogmag
# from FLAGS import PARAM
# import os
# from pypesq import pesq
import numpy as np
if __name__ == "__main__":
  wave,sr = audio_tool.read_audio('exp/rnn_speech_enhancement/decode_nnet_C002_2/nnet_C002_2_003_s_8_21_MIX_3_factory_8k.wav') # s_8_21_MIX_3_factory_8k
  wave_spec = spectrum_tool.magnitude_spectrum_librosa_stft(wave,512,256)
  wave_spec = np.log(wave_spec+1e3)-np.log(1e3)
  wave_spec = np.transpose(wave_spec,[1,0])
  pre_frame = 0
  subs_spec = []
  for i in range(np.shape(wave_spec)[0]):
    subs_spec.append(wave_spec[i]-pre_frame)
    pre_frame = wave_spec[i]

  subs_spec = np.array(subs_spec)
  spectrum_tool.picture_spec(subs_spec.T,"233_en2_2")

  # audio_tool.gen_mixed_wav('exp/real_test_fair/ITU_T_Test/refer_wav',
  #                          'exp/real_test_fair/7_noise',
  #                          'exp/real_test_fair/ITU_T_Test/mixed_wav',force=True)

  # ref, sr = audio_tool.read_audio('test_cc/ref.wav')
  # cleaned, sr = audio_tool.read_audio('test_cc/cleaned.wav')
  # mixed, sr = audio_tool.read_audio('test_cc/mixed.wav')
  # print(sr)
  # print(ref)
  # print(cleaned)
  # print(pesq(cleaned,ref,sr))

  # print(pesq(ref,mixed,sr))
  # A = tf.constant([[[1,1],[1,1]],[[1,1],[1,1]]])
  # B = tf.constant([
  #   [[2, 2], [2, 2]]
  #                 ])
  # print(np.shape(A),np.shape(B))
  # C = tf.matmul(A,B)
  # sess = tf.Session()
  # print(sess.run(
  #   C
  # ))
  # print(tf.Session().run(tf.nn.softmax([[[3.0],[1.0]],[[3.0],[1.0]]],axis=-2)))

  # waveData, sr = audio_tool.read_audio('exp/rnn_speech_enhancement/speech0_8k.wav')
  # x_spec_t = spectrum_tool.magnitude_spectrum_librosa_stft(waveData,
  #                                                          PARAM.NFFT,
  #                                                          PARAM.OVERLAP)
  # new_mag = rm_norm_mag_spec(normedLogmag2normedMag(norm_logmag_spec(x_spec_t,0.00001),0.00001))
  # new_mag = tf.Session().run(new_mag)
  # reY = spectrum_tool.griffin_lim(new_mag,
  #                                 PARAM.NFFT,
  #                                 PARAM.OVERLAP,
  #                                 PARAM.GRIFFIN_ITERNUM,
  #                                 waveData)
  # audio_tool.write_audio('exp/rnn_speech_enhancement/speech0_8k_new.wav',
  #                        reY,
  #                        sr)

