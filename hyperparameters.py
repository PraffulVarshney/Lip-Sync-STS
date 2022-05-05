from tensorflow.contrib.training import HParams
from glob import glob
import os, pickle

# Default hyperparameters
# Hparams Class to hold a set of hyperparameters as name-value pairs
hyperparameters = HParams(
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	rescale=True,  # rescaling the pitch and time of audio prior to preprocessing
	rescaling_max=0.9,  # Rescaling value
	max_mel_frames=900,
	use_lws=False,

	n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
	hop_size=200,  # 200 = 12.5 ms (0.0125 * sample_rate) number of samples between each successive FFT window
	win_size=800,  # 800 = 50 ms(0.05 * sample_rate)
	sample_rate=16000,  # 16000Hz (corresponding to librispeech)   number of samples of audio recorded every second.

	frame_shift_ms=None,

	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,  # normalize mel spectrograms to some predefined range
	allow_clipping_in_normalization=True,
	symmetric_mels=True,
	max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
	normalize_for_wavenet=True,  # to rescale to [0, 1] for wavenet.
	clip_for_wavenet=True,
	preemphasize=True,  # whether to apply filter
	preemphasis=0.97,  # filter coefficient.

	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,  # 55 for male and 95 for female ( Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,

	# Griffin Lim
	power=1.5,
	griffin_lim_iters=60,  # Number of G&L iterations, 60 to ensure convergence.
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)
