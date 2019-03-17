# -*- coding: utf-8 -*-
"""
@author: PengChuan
这一部分是语音信号的评价指标，用来评估语音信号降噪的质量，判断结果好坏
    pesq：perceptual evaluation of speech quality，语音质量听觉评估
    stoi：short time objective intelligibility，短时客观可懂度，尤其在低SNR下，可懂度尤其重要
    ssnr: segmental SNR，分段信噪比(时域指标)，它是参考信号和信号差的比值，衡量的是降噪程度
"""

import os
import tempfile
import numpy as np
import librosa
import platform
from utils import audio_tool

# 导入pesq.exe，用于语音质量的评估
PESQ_PATH = os.path.split(os.path.realpath(__file__))[0]
if 'Linux' in platform.system():
    PESQ_PATH = os.path.join(PESQ_PATH,
                             'pesq.ubuntu16.exe',
                             #  'pesq.exe',
                             )
else:
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.win10.exe')

# Machine limits for integer types
# Maximum value of given dtype
max_int = np.iinfo(np.int16).max
# (float) The smallest positive usable number.
# Type of tiny is an appropriate floating point type.
min_pf = np.finfo(np.float32).tiny

# 根据人耳对信噪比有意义的范围，做了对过高或过低信噪比的限制
ssnr_min = -40
ssnr_max = 40


def calc_pesq(ref_sig, deg_sig, samplerate, is_file=False):
    '''
    计算语音质量听觉评估
    return 评估的分数，分数高的结果比较好
    '''
    if 'Windows' in platform.system():
        # 暂不支持windows下pesq计算
        return 0

    if is_file:
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, ref_sig, deg_sig))
        msg = output.read()
    else:
        tmp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        tmp_deg = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        # librosa.output.write_wav(tmp_ref.name, ref_sig, samplerate)
        # librosa.output.write_wav(tmp_deg.name, deg_sig, samplerate)
        audio_tool.write_audio(tmp_ref.name,ref_sig,samplerate)
        audio_tool.write_audio(tmp_deg.name,deg_sig,samplerate)
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, tmp_ref.name, tmp_deg.name))
        msg = output.read()
        # print(msg)
        tmp_ref.close()
        tmp_deg.close()
        # os.unlink(tmp_ref.name)
        # os.unlink(tmp_deg.name)
    # print(msg)
    score = msg.split('Prediction : PESQ_MOS = ')
    # print(msg)
    # exit(0)
    # print(score)
    if len(score)<=1:
      print('error')
      return 2.0
    return float(score[1][:-1])

'''

def calc_ssnr(ref_sig, deg_sig, frame_size, mid_only=False):
    # 计算分段信噪比
    ref_frame = sdcore.frame(ref_sig, frame_size, frame_size, center=False)
    deg_frame = sdcore.frame(deg_sig, frame_size, frame_size, center=False)
    if mid_only:
        i = len(ref_frame) // 2
        ref_frame = ref_frame[i, :]
        deg_frame = deg_frame[i, :]
    noise_frame = ref_frame - deg_frame
    ref_energy = np.sum(ref_frame ** 2, axis=-1) + min_pf
    noise_energy = np.sum(noise_frame ** 2, axis=-1) + min_pf
    ssnr = 10 * np.log10(ref_energy / noise_energy)
    if mid_only:
        # return min(ssnr_max, max(ssnr_min, ssnr))
        return ssnr
    else:
        ssnr[ssnr < ssnr_min] = ssnr_min
        ssnr[ssnr > ssnr_max] = ssnr_max
        return np.mean(ssnr)
'''
