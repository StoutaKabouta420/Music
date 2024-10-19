import librosa
import soundfile as sf

class AudioInput:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_time_series = None
        self.sample_rate = None

    def load_audio(self):
        """Load the audio file and return the time series and sample rate."""
        try:
            # librosa can handle MP3 files
            self.audio_time_series, self.sample_rate = librosa.load(self.file_path, sr=None)
            return self.audio_time_series, self.sample_rate
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def get_duration(self):
        """Get the duration of the audio in seconds."""
        if self.audio_time_series is None:
            self.load_audio()
        if self.audio_time_series is not None:
            return librosa.get_duration(y=self.audio_time_series, sr=self.sample_rate)
        return None

    def get_tempo(self):
        """Estimate the tempo of the audio."""
        if self.audio_time_series is None:
            self.load_audio()
        if self.audio_time_series is not None:
            tempo, _ = librosa.beat.beat_track(y=self.audio_time_series, sr=self.sample_rate)
            return float(tempo)  # Convert numpy.float64 to Python float
        return None

    def get_spectral_centroid(self):
        """Compute the spectral centroid."""
        if self.audio_time_series is None:
            self.load_audio()
        if self.audio_time_series is not None:
            spectral_centroids = librosa.feature.spectral_centroid(y=self.audio_time_series, sr=self.sample_rate)[0]
            return float(np.mean(spectral_centroids))  # Convert numpy.float64 to Python float
        return None

    def get_zero_crossing_rate(self):
        """Compute the zero crossing rate."""
        if self.audio_time_series is None:
            self.load_audio()
        if self.audio_time_series is not None:
            zero_crossings = librosa.feature.zero_crossing_rate(y=self.audio_time_series)[0]
            return float(np.mean(zero_crossings))  # Convert numpy.float64 to Python float
        return None