#include <algorithm>
#include <world/stonemask.h>
#include <world/dio.h>
#include <stdlib.h>
#include <vector>

/*
* DIO F0 extraction algorithm.

		Parameters
		----------
		x : ndarray
			Input waveform signal.
		fs : int
			Sample rate of input signal in Hz.
		f0_floor : float
			Lower F0 limit in Hz.
			Default: 71.0
		f0_ceil : float
			Upper F0 limit in Hz.
			Default: 800.0
		channels_in_octave : float
			Resolution of multiband processing; normally shouldn't be changed.
			Default: 2.0
		frame_period : float
			Period between consecutive frames in milliseconds.
			Default: 5.0
		speed : int
			The F0 estimator may downsample the input signal using this integer factor
			(range [1;12]). The algorithm will then operate on a signal at fs/speed Hz
			to reduce computational complexity, but high values may negatively impact
			accuracy.
			Default: 1 (no downsampling)
		allowed_range : float
			Threshold for voiced/unvoiced decision. Can be any value >= 0, but 0.02 to 0.2
			is a reasonable range. Lower values will cause more frames to be considered
			unvoiced (in the extreme case of `threshold=0`, almost all frames will be unvoiced).
			Default: 0.1

		Returns
		-------
		f0 : ndarray
			Estimated F0 contour.
		temporal_positions : ndarray
			Temporal position of each frame.
*/
void func_get_dio(
	std::vector<double> x,
	int fs,
	float f0_floor,
	float f0_ceil,
	float channels_in_octave,
	float frame_period,
	int speed,
	float allowed_range,
	std::vector<double>* f0,
	std::vector<double>* temporal_positions
) {
	/*
	cdef int x_length = <int>len(x)
	cdef DioOption option
	InitializeDioOption(&option)
	option.channels_in_octave = channels_in_octave
	option.f0_floor = f0_floor
	option.f0_ceil = f0_ceil
	option.frame_period = frame_period
	option.speed = speed
	option.allowed_range = allowed_range
	f0_length = GetSamplesForDIO(fs, x_length, option.frame_period)
	cdef np.ndarray[double, ndim=1, mode="c"] f0 = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	cdef np.ndarray[double, ndim=1, mode="c"] temporal_positions = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	Dio(&x[0], x_length, fs, &option, &temporal_positions[0], &f0[0])
	return f0, temporal_positions
	*/
	int x_length = x.size();
	DioOption option;
	InitializeDioOption(&option);
	option.channels_in_octave = channels_in_octave;
	option.f0_floor = f0_floor;
	option.f0_ceil = f0_ceil;
	option.frame_period = frame_period;
	option.speed = speed;
	option.allowed_range = allowed_range;
	int f0_length = GetSamplesForDIO(fs, x_length, option.frame_period);
	f0->resize(f0_length);
	temporal_positions->resize(f0_length);
	Dio(x.data(), x_length, fs, &option, temporal_positions->data(), f0->data());
}


/*
* StoneMask F0 refinement algorithm.

		Parameters
		----------
		x : ndarray
			Input waveform signal.
		f0 : ndarray
			Input F0 contour.
		temporal_positions : ndarray
			Temporal positions of each frame.
		fs : int
			Sample rate of input signal in Hz.

		Returns
		-------
		refined_f0 : ndarray
			Refined F0 contour.
*/
std::vector<double> func_stonemask(
	std::vector<double> x,
	std::vector<double>* f0,
	std::vector<double>* temporal_positions,
	int fs
) {
	/*
	cdef int x_length = <int>len(x)
	cdef int f0_length = <int>len(f0)
	cdef np.ndarray[double, ndim=1, mode="c"] refined_f0 = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	StoneMask(&x[0], x_length, fs, &temporal_positions[0],
		&f0[0], f0_length, &refined_f0[0])
	return refined_f0
	*/
	int x_length = x.size();
	int f0_length = f0->size();
	std::vector<double>refined_f0(f0_length);
	StoneMask(x.data(), x_length, fs, temporal_positions->data(), f0->data(), f0_length, refined_f0.data());
	return refined_f0;
}

std::vector<int64_t> func_prepare_f0(
	std::vector<double> dSamples,
	int iSamplerate,
	int iHiddentUnitNum,
	int iPitch
) {
	/*
		def transcribe(self, source, sr, length, transform):
			feature_pit = self.feature_input.compute_f0(source, sr)
			feature_pit = feature_pit * 2 ** (transform / 12)
			feature_pit = resize2d_f0(feature_pit, length)
			coarse_pit = self.feature_input.coarse_f0(feature_pit)
			return coarse_pit
	*/

	// why 32000?
	int sr = 32000;
	int iHop = 320;
	float fDefaultF0Floor = 71.f;
	float channels_in_octave = 2.f;
	float frame_period = 1000.f * iHop / sr;
	int speed = 1;
	float allowed_range = 0.1f;
	int iF0Ceil = 800;
	std::vector<double> f0;
	std::vector<double> t;
	func_get_dio(dSamples, sr, fDefaultF0Floor, iF0Ceil, channels_in_octave, frame_period, speed, allowed_range, &f0, &t);
	f0 = func_stonemask(dSamples, &f0, &t, sr);
	std::transform(
		f0.begin(),
		f0.end(),
		f0.begin(),
		[iPitch](double i) {
			return i * pow(2, 1.f * iPitch / 12.f);
		});
	// feature_pit = resize2d_f0(feature_pit, length) ????,Dio的sr修正后是否不需要这步？
	// 计算coarse_f0
	double f0_min = 1127.f * log(1 + 50.f / 700.f);
	double f0_max = 1127.f * log(1 + 1100.f / 700.f);
	/*
	f0_mel = 1127 * np.log(1 + f0 / 700)
	f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
	f0_mel[f0_mel <= 1] = 1
	f0_mel[f0_mel > 255] = 255
	f0_coarse = np.rint(f0_mel).astype(np.int)
	*/
	std::transform(
		f0.begin(),
		f0.end(),
		f0.begin(),
		[f0_min, f0_max](double i) {
			i = 1127 * log(1 + i / 700.f);
			if (i > 0) {
				i = (i - f0_min) * 254.f / (f0_max - f0_min) + 1;
			}
			if (i <= 1) {
				i = 1;
			}
			if (i > 255) {
				i = 255;
			}
			i = rint(i);
			return i;
		});
	std::vector<int64_t> iF0(f0.size());
	for (int i = 0; i < f0.size(); i++) {
		iF0[i] = f0[i];
	}
	return iF0;
}