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
#     stfts = tf.signal.stft(audio_ts, frame_length=c.WIN_SIZE,
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
#     audio_ts = load_wav(sample_path)
#
#     # Remove the first and last second
#     audio_ts = audio_ts[RAV_SR:-RAV_SR]
#     duration_diff = 193794 - audio_ts.shape[0]
#     audio_ts = np.pad(audio_ts, pad_width=(0, duration_diff),
#                       mode='constant', constant_values=0)
#     melspecgram = sgh.wave_to_melspecgram(audio_ts)
#     print(sample_filename)
#     melspecgram = sgh.scale_melspecgram(melspecgram)
#     # print(np.amin(melspecgram), np.amax(melspecgram))
#     # print(melspecgram.shape)
#     plt.subplot(4, 4, i)
#     plt.pcolormesh(melspecgram)
#     i += 1
