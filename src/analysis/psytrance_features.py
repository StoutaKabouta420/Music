import librosa
import numpy as np

class PsytranceFeatureExtractor:
    def __init__(self, audio, sr):
        self.sr = sr
        # Limit to 1-3 minutes (or less if the track is shorter)
        min_duration = 60  # 1 minute
        max_duration = 3 * 60  # 3 minutes
        max_samples = min(len(audio), max_duration * sr)
        min_samples = min(max_samples, min_duration * sr)
        self.audio = audio[min_samples:max_samples]
        self.duration = len(self.audio) / sr

    def extract_features(self):
        features = {}
        
        # Rhythm features
        tempo, beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        features['tempo'] = tempo

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
        features['spectral_centroids'] = np.mean(spectral_centroids)

        # MFCC
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
        features['mfccs'] = np.mean(mfccs, axis=1)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=self.audio, sr=self.sr)
        features['chroma'] = np.mean(chroma, axis=1)

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
        features['onset_strength'] = np.mean(onset_env)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
        features['spectral_contrast'] = np.mean(contrast, axis=1)

        # Add duration as a feature
        features['duration'] = self.duration

        return features