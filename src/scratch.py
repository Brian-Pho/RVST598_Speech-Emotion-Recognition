"""
This file holds useful functions to refer back to during programming. Functions
in this file aren't runnable but should help debugging or viewing data.
"""

# # Displaying and playing a waveform
# file_num = 79
# data = processed_samples[file_num][0]
# data /= np.amax(data)
# sd.play(data, processed_samples[file_num][1], blocking=True)
# plt.figure()
# librosa.display.waveplot(data, sr=processed_samples[file_num][1])
# plt.show()

# # Displaying a spectrogram
# plt.pcolormesh(t, f, amplitude)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# # Create log-mel spectrogram from TensorFlow
# sess = tf.compat.v1.Session()
# with sess.as_default():
#     stfts = tf.signal.stft(wav, frame_length=c.WIN_SIZE,
#                            frame_step=c.STEP_SIZE)
#     spectrograms = tf.abs(stfts)
#     num_spectrogram_bins = stfts.shape[-1].value
#     lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 20000.0, 100
#     linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#         num_mel_bins, num_spectrogram_bins, c.SR, lower_edge_hertz,
#         upper_edge_hertz)
#     mel_spectrograms = tf.tensordot(
#         spectrograms, linear_to_mel_weight_matrix, 1)
#     mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
#         linear_to_mel_weight_matrix.shape[-1:]))
#     log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
#     log_mel_spectrograms = log_mel_spectrograms.eval()
# # print(spectrograms.shape)
# log_mel_spectrograms = np.swapaxes(log_mel_spectrograms, 0, 1)
# # log_mel_spectrograms /= np.abs(log_mel_spectrograms)
# plt.pcolormesh(log_mel_spectrograms)
# plt.show()

# # Displaying multiple plots
# i = 1
# for sample_filename in os.listdir(actor_path)[:16]:
#     sample_path = os.path.join(actor_path, sample_filename)
#
#     # Read the sample
#     wav = load_wav(sample_path)
#
#     # Remove the first and last second
#     wav = wav[RAV_SR:-RAV_SR]
#     duration_diff = 193794 - wav.shape[0]
#     wav = np.pad(wav, pad_width=(0, duration_diff),
#                       mode='constant', constant_values=0)
#     melspecgram = sg.wave_to_melspecgram(wav)
#     print(sample_filename)
#     melspecgram = sg.scale_melspecgram(melspecgram)
#     # print(np.amin(melspecgram), np.amax(melspecgram))
#     # print(melspecgram.shape)
#     plt.subplot(4, 4, i)
#     plt.pcolormesh(melspecgram)
#     i += 1
# plt.show()

# # Change matplotlib colors
# plt.pcolormesh(melspecgram, cmap=plt.get_cmap("YlGnBu"))
# plt.colorbar(melspecgram)
# plt.show()

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
#
# # Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(3, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(3, size=(100, 1))
#
# model = Sequential()
# model.add(Dense(64, input_dim=20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)
# print(score)
# prediction = model.predict(x_test[0:5])
# print(prediction, y_test[0:5])

# # Confusion matrix example
# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
#
# array = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
#          [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
#          [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
#          [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
#          [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
#          [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
#          [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]
# df_cm = pd.DataFrame(array, index=[i for i in "ABCDEFGHIJK"],
#                      columns=[i for i in "ABCDEFGHIJK"])
# plt.figure(figsize=(10, 7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
