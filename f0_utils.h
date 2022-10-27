#pragma once
void func_get_dio(std::vector<double> x, int fs, float f0_floor, float f0_ceil, float channels_in_octave, float frame_period, int speed, float allowed_range, std::vector<double>* f0, std::vector<double>* temporal_positions);

std::vector<double> func_stonemask(std::vector<double> x, std::vector<double>* f0, std::vector<double>* temporal_positions, int fs);

std::vector<int64_t> func_prepare_f0(std::vector<double> dSamples, int iSamplerate, int iHiddentUnitNum, int iPitch);

