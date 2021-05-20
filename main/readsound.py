import os
import wave

import pandas as pandas
import tensorflow as tf
import numpy as np
from numpy import tile, newaxis, dot, sinc
from scipy.io import wavfile

fname = "yes"
# sampling_rate = tf.constant(7812)
init_sampling_rate = 7812
dest_sampling_rate = 16000


# wavfile = "../wavfile/silence.wav"

# with tf.Session() as sess:
# Returns a tuple of Tensor objects (audio, sample_rate).
# raw_audio = tf.io.read_file(wavfile)
# input_signal, sampling_rate = tf.audio.decode_wav(raw_audio)

# ! /usr/bin/python
#
# Author: Gaute Hope (gaute.hope@nersc.no) / 2015
#
# based on example from matlab sinc function and
# interpolate.m by H. Hobæk (1994).
#
# this implementation is similar to the matlab sinc-example, but
# calculates the values sequentially and not as a single matrix
# matrix operation for all the values.
#
def resample(x, k):
    """
  Resample the signal to the given ratio using a sinc kernel
  input:
    x   a vector or matrix with a signal in each row
    k   ratio to resample to
    returns
    y   the up or downsampled signal
    when downsampling the signal will be decimated using scipy.signal.decimate
  """

    if k < 1:
        raise NotImplementedError('downsampling is not implemented')

    if k == 1:
        return x  # nothing to do

    return upsample(x, k)


def upsample(x, k):
    """
  Upsample the signal to the given ratio using a sinc kernel
  input:
    x   a vector or matrix with a signal in each row
    k   ratio to resample to
    returns
    y   the up or downsampled signal
    when downsampling the signal will be decimated using scipy.signal.decimate
  """

    assert k >= 1, 'k must be equal or greater than 1'

    mn = x.shape
    if len(mn) == 2:
        m = mn[0]
        n = mn[1]
    elif len(mn) == 1:
        m = 1
        n = mn[0]
    else:
        raise ValueError("x is greater than 2D")

    nn = int(n * k)

    xt = np.linspace(1, n, n)
    xp = np.linspace(1, n, nn)

    return interp(xp, xt, x)


def interp(xp, xt, x):
    """
  Interpolate the signal to the new points using a sinc kernel
  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  output:
  y     the interpolated signal at points xp
  """

    mn = x.shape
    if len(mn) == 2:
        m = mn[0]
        n = mn[1]
    elif len(mn) == 1:
        m = 1
        n = mn[0]
    else:
        raise ValueError("x is greater than 2D")

    nn = len(xp)

    y = np.zeros((m, nn))

    for (pi, p) in enumerate(xp):
        si = np.tile(np.sinc(xt - p), (m, 1))
        y[:, pi] = np.sum(si * x)

    return y.squeeze()

def main_generate_tensor():
    for i in range(0, 7):
        input_raw = []
        with open("../tensor/" + fname + str(i), 'r') as f:
            for line in f:
                input_raw.append([float(line)])
        if init_sampling_rate > 9000:
            continue
        # if sampling_rate < 9000:  # make it to 16000 sampling rate
        time = 1 / init_sampling_rate
        # upscale = np.array([dest_sampling_rate/sampling_rate])
        # x_array = np.arange(0, 1, 1/sampling_rate)
        val_array = np.array(np.array(input_raw).flatten())
        upscale = dest_sampling_rate / init_sampling_rate
        # upscale = np.arange(0, 1, 1/dest_sampling_rate)
        result = resample(val_array, upscale)
        result = np.array(result)
        # result_np = []
        # for v in result:
        #     result_np.append([v])

        # raw_audio = tf.io.read_file("../wavfile/017c4098_nohash_1.wav")
        # input_signal, sampling_rate = tf.audio.decode_wav(raw_audio)
        # print(input_signal.get_shape())

        #float to pcm
        #some workaround due to tensorflow bug
        #speech command seems to have 16 bit data, whereas the encode_wav function from tf only supports float32
        result *= 32767 #convert to 16bit int

        wavfile.write("../wavfile/" + fname + str(i) + ".wav", dest_sampling_rate, result.astype(np.int16))
        # input_signal = tf.constant(result_np, dtype=tf.float32)  # shape (audio values, channels)
        # print(input_signal.get_shape())

        # Returns a Tensor of type string.
        # output_signal = tf.audio.encode_wav(input_signal, tf.constant(dest_sampling_rate), name=fname + str(i))
        # tf.print(output_signal, output_stream="file://../wavfile/" + fname + str(i) + ".wav")  # , output_stream=sys.stderr)


def compare_header_and_size(wav_filename):
    with wave.open(wav_filename, 'r') as fin:
        header_fsize = (fin.getnframes() * fin.getnchannels() * fin.getsampwidth()) + 44
    file_fsize = os.path.getsize(wav_filename)
    return header_fsize != file_fsize

# res= compare_header_and_size("../wavfile/yes0.wav")
# print('The following files are corrupted:')
# print(res)

main_generate_tensor()

# input_signal
# val = tf.squeeze(input_signal, axis=-1)


# with open("testwav.txt", 'w') as f:
#     arr_ss= input_signal.numpy()
#     for v in arr_ss:
#         f.write(str(v))
#         f.write("\n")
