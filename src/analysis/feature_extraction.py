import librosa
import numpy as np

class AudioFeatures:
    def __init__(self, audio_time_series, sample_rate):
        self.audio = audio_time_series
        self.sr = sample_rate

    def get_tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        return tempo

    def get_spectral_centroid(self):
        return np.mean(librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0])

    def get_zero_crossing_rate(self):
        return np.mean(librosa.feature.zero_crossing_rate(y=self.audio)[0])

    def get_mfcc(self, n_mfcc=13):
        return np.mean(librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=n_mfcc), axis=1)

    def get_chroma_features(self):
        return np.mean(librosa.feature.chroma_stft(y=self.audio, sr=self.sr), axis=1)

    def get_spectral_contrast(self):
        return np.mean(librosa.feature.spectral_contrast(y=self.audio, sr=self.sr), axis=1)

    def get_tonnetz(self):
        return np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio), sr=self.sr), axis=1)

    def get_all_features(self):
        return {
            'tempo': self.get_tempo(),
            'spectral_centroid': self.get_spectral_centroid(),
            'zero_crossing_rate': self.get_zero_crossing_rate(),
            'mfcc': self.get_mfcc(),
            'chroma': self.get_chroma_features(),
            'spectral_contrast': self.get_spectral_contrast(),
            'tonnetz': self.get_tonnetz()
        }