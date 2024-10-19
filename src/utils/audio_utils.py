import os

def is_audio_file(filename):
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg']
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

def get_audio_files(directory):
    return [f for f in os.listdir(directory) if is_audio_file(f)]