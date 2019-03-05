# import soundfile
import librosa
import numpy as np
import FLAGS

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''


def read_audio(file):
  data, sr = librosa.load(file)
  data = librosa.resample(data, sr, FLAGS.PARAM.FS)
  return data*32767, sr


def write_audio(file, data, sr):
  data /= 32767
  return librosa.output.write(file, data, sr)
