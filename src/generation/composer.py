import numpy as np
import librosa
import soundfile as sf
from scipy import signal

class AudioComposer:
    def __init__(self, original_audio, sr, transformed_features):
        self.original_audio = original_audio
        self.sr = sr
        self.transformed_features = transformed_features

    def simple_time_stretch(self, audio, rate):
        """A simple time stretching method using resampling"""
        return signal.resample(audio, int(len(audio) / rate))

    def apply_filter(self, audio, cutoff, btype='lowpass'):
        """Apply a low-pass or high-pass filter"""
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(6, normal_cutoff, btype=btype, analog=False)
        return signal.filtfilt(b, a, audio)

    def compose(self):
        # Extract the transformed features
        new_tempo = self.transformed_features[0]
        new_spectral_centroid = self.transformed_features[1]
        new_zero_crossing_rate = self.transformed_features[2]

        # Time stretch to match new tempo
        original_tempo, _ = librosa.beat.beat_track(y=self.original_audio, sr=self.sr)
        tempo_ratio = original_tempo / new_tempo
        y_stretched = self.simple_time_stretch(self.original_audio, tempo_ratio)

        # Pitch shift to approximate new spectral centroid
        original_centroid = np.mean(librosa.feature.spectral_centroid(y=self.original_audio, sr=self.sr))
        pitch_shift = 12 * np.log2(new_spectral_centroid / original_centroid)
        y_shifted = librosa.effects.pitch_shift(y=y_stretched, sr=self.sr, n_steps=pitch_shift)

        # Apply a simple low-pass or high-pass filter to approximate zero crossing rate
        original_zcr = np.mean(librosa.feature.zero_crossing_rate(y=self.original_audio)[0])
        if new_zero_crossing_rate > original_zcr:
            y_filtered = self.apply_filter(y_shifted, cutoff=1000, btype='highpass')
        else:
            y_filtered = self.apply_filter(y_shifted, cutoff=1000, btype='lowpass')

        return y_filtered

    def save_audio(self, y, filename):
        sf.write(filename, y, self.sr)