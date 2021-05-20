import tensorflow as tf

wavfile = "../wavfile/yes0.wav"
# with open("../wavfile/yes0.wav")
# Returns a tuple of Tensor objects (audio, sample_rate).
raw_audio = tf.io.read_file(wavfile)
input_signal, sampling_rate = tf.audio.decode_wav(raw_audio)
print(len(input_signal))