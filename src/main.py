import os
import numpy as np
from audio.input import AudioInput
from analysis.feature_extraction import AudioFeatures
from ml.model import StyleTransfer

def create_feature_vector(features):
    vector = []
    for f in features:
        if isinstance(features[f], (np.ndarray, list)):
            vector.extend(features[f])
        elif isinstance(features[f], (int, float, np.float64)):
            vector.append(features[f])
    return np.array(vector)

def main():
    # Specify the paths to your input audio files
    input_file1 = os.path.join('data', 'input', '130493__electrobuz__drumstep_loop_180bpm.wav')
    input_file2 = os.path.join('data', 'input', '252809__scydan__breakbeat-180bpm.wav')  # Replace with another actual file name

    # Create instances of AudioInput
    audio_input1 = AudioInput(input_file1)
    audio_input2 = AudioInput(input_file2)

    # Load the audio files
    audio1, sr1 = audio_input1.load_audio()
    audio2, sr2 = audio_input2.load_audio()

    if audio1 is None or audio2 is None:
        print("Failed to load one or both audio files")
        return

    # Extract features
    features1 = AudioFeatures(audio1, sr1).get_all_features()
    features2 = AudioFeatures(audio2, sr2).get_all_features()

    # Prepare feature vectors
    feature_vector1 = create_feature_vector(features1)
    feature_vector2 = create_feature_vector(features2)

    print(f"Feature vector 1 shape: {feature_vector1.shape}")
    print(f"Feature vector 2 shape: {feature_vector2.shape}")

    # Initialize and fit style transfer model
    st_model = StyleTransfer()
    try:
        st_model.fit(np.vstack([feature_vector1, feature_vector2]))
    except ValueError as e:
        print(f"Error fitting the model: {e}")
        return

    # Perform style transfer
    try:
        transformed_features = st_model.transform(feature_vector1, feature_vector2, alpha=0.3)
    except Exception as e:
        print(f"Error during style transfer: {e}")
        return

    print(f"\nAudio file 1: {input_file1}")
    print(f"Audio file 2: {input_file2}")
    print(f"\nOriginal features of song1 (first 5): {feature_vector1[:5]}")
    print(f"Style features of song2 (first 5): {feature_vector2[:5]}")
    print(f"Transformed features (first 5): {transformed_features[0][:5]}")

    # TODO: Implement audio synthesis from transformed features

if __name__ == "__main__":
    main()