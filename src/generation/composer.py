import numpy as np
from scipy import signal
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioComposer:
    def __init__(self, transformed_features, sr=22050):
        self.transformed_features = transformed_features
        self.sr = sr  # Sample rate
        self.duration = min(max(120, self.transformed_features[-1]), 300)  # Between 120 and 300s
        self.chunk_size = 5  # Process in 5-second chunks
        logging.info(f"Initialized AudioComposer with duration: {self.duration:.2f}s, sr: {self.sr}")

    def generate_base_audio(self):
        """Generates the basic waveform with bassline, melody, drums, and harmonics."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration), False)
        waveform = np.zeros_like(t, dtype=np.float32)

        # Bassline (low frequencies)
        bass_freq = 50 + abs(self.transformed_features[0]) * 50
        bass_wave = 0.4 * np.sin(2 * np.pi * bass_freq * t)
        waveform += bass_wave

        # Melody (mid frequencies)
        melody_freq = 200 + abs(self.transformed_features[1]) * 200
        melody_wave = 0.3 * np.sin(2 * np.pi * melody_freq * t) + 0.2 * np.sin(4 * np.pi * melody_freq * t)
        waveform += melody_wave

        # Rhythmic element (e.g., drums)
        drum_pattern = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 5)) % 0.5
        waveform += drum_pattern * 0.2

        # Harmonics to add richness
        waveform += 0.1 * np.sin(2 * np.pi * (melody_freq * 2) * t)
        waveform += 0.05 * np.sin(2 * np.pi * (melody_freq * 3) * t)

        # Normalize waveform to prevent clipping
        waveform /= np.max(np.abs(waveform))
        
        logging.info(f"Generated base audio with shape: {waveform.shape}, duration: {len(waveform)/self.sr:.2f}s")
        return waveform

    def apply_filter(self, audio, cutoff, btype='lowpass'):
        """Applies a Butterworth filter to the audio."""
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(6, normal_cutoff, btype=btype, analog=False)
        filtered = signal.lfilter(b, a, audio)
        logging.info(f"Applied {btype} filter with cutoff {cutoff}")
        return filtered

    def simple_time_stretch(self, audio, rate):
        """Stretches or compresses the audio by a specified rate."""
        rate = np.clip(rate, 0.5, 2.0)  # Limit stretching/compression
        input_length = len(audio)
        output_length = int(input_length / rate)
        stretched = signal.resample(audio, output_length)
        logging.info(f"Time stretched audio from {input_length} to {output_length} samples, rate: {rate:.2f}")
        return stretched

    def simple_pitch_shift(self, audio, n_steps):
        """Shifts the pitch of the audio by a specified number of semitones."""
        n_steps = np.clip(n_steps, -12, 12)  # Limit pitch shift
        factor = 2 ** (n_steps / 12)
        stretched = self.simple_time_stretch(audio, factor)
        shifted = signal.resample(stretched, len(audio))
        logging.info(f"Pitch shifted audio by {n_steps:.2f} steps")
        return shifted

    def add_effects(self, audio):
        """Applies delay, reverb, and distortion to the audio."""
        # Simple delay effect
        delay_time = int(self.sr * 0.25)  # 250ms delay
        delayed = np.zeros_like(audio)
        delayed[delay_time:] = audio[:-delay_time]
        audio = audio + 0.3 * delayed

        # Add reverb (simple convolution-based reverb)
        reverb_amount = 0.2
        reverb_kernel = np.exp(-np.linspace(0, 2, int(self.sr * 0.5)))
        reverb = np.convolve(audio, reverb_kernel, mode='full')

        # Adjust the length of the reverb to match the original audio length
        if len(reverb) > len(audio):
            reverb = reverb[:len(audio)]
        else:
            reverb = np.pad(reverb, (0, len(audio) - len(reverb)))

        audio = audio + reverb_amount * reverb

        # Distortion effect
        audio = np.tanh(audio * 2) * 0.5

        return audio


    def process_chunk(self, chunk, tempo, spectral_centroid, zero_crossing_rate):
        """Processes a chunk of audio with tempo adjustments, filtering, and effects."""
        tempo_factor = np.clip(tempo / 120, 0.5, 2.0)  # Limit tempo change
        time_stretched = self.simple_time_stretch(chunk, tempo_factor)
        pitch_shift = np.clip(12 * np.log2(spectral_centroid / 1000), -12, 12)
        pitch_shifted = self.simple_pitch_shift(time_stretched, pitch_shift)

        # Apply filter based on zero-crossing rate
        filter_type = 'highpass' if zero_crossing_rate > 0.5 else 'lowpass'
        filtered = self.apply_filter(pitch_shifted, cutoff=1000, btype=filter_type)

        return self.add_effects(filtered)

    def compose(self):
        """Composes the final audio by processing each chunk."""
        base_audio = self.generate_base_audio()

        # Extracted features for composition parameters
        tempo = np.clip(120 + self.transformed_features[0] * 10, 60, 180)  # Limit tempo range
        spectral_centroid = np.clip(1000 + abs(self.transformed_features[1]) * 1000, 500, 5000)
        zero_crossing_rate = np.clip(self.transformed_features[2], 0, 1)

        logging.info(f"Composition parameters: Tempo: {tempo:.2f}, Spectral Centroid: {spectral_centroid:.2f}, ZCR: {zero_crossing_rate:.2f}")

        # Process chunks of the base audio
        result = []
        for i in range(0, len(base_audio), self.chunk_size * self.sr):
            chunk = base_audio[i:i + self.chunk_size * self.sr]
            if len(chunk) > 0:
                processed_chunk = self.process_chunk(chunk, tempo, spectral_centroid, zero_crossing_rate)
                result.append(processed_chunk)
                logging.info(f"Processed chunk {i//self.sr//self.chunk_size + 1}, shape: {processed_chunk.shape}")

        if result:
            composed_audio = np.concatenate(result)
            # Ensure the final audio has the correct duration
            composed_audio = signal.resample(composed_audio, int(self.duration * self.sr))
            logging.info(f"Final composed audio shape: {composed_audio.shape}, duration: {len(composed_audio)/self.sr:.2f}s")
            return composed_audio
        else:
            logging.warning("No audio data generated")
            return np.array([])

    def save_audio(self, y, filename):
        """Saves the composed audio to a file."""
        if len(y) > 0:
            # Normalize audio
            y = y / np.max(np.abs(y))
            sf.write(filename, y, self.sr)
            logging.info(f"Audio saved to {filename} with duration: {len(y)/self.sr:.2f}s")
        else:
            logging.warning("No audio data to save.")
