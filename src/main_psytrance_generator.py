import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from analysis.psytrance_features import PsytranceFeatureExtractor
from ml.psytrance_vae import PsytranceVAE, vae_loss
from audio.input import AudioInput
from generation.composer import AudioComposer
from utils.audio_utils import get_audio_files
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_feature_vector(features):
    vector = []
    for f in features:
        if isinstance(features[f], (np.ndarray, list)):
            vector.extend(features[f])
        elif isinstance(features[f], (int, float, np.float64)):
            vector.append(features[f])
    return np.array(vector)

def main():
    psytrance_dir = 'data/psytrance_samples/'
    features_list = []
    sr = None
    
    for file in get_audio_files(psytrance_dir):
        file_path = os.path.join(psytrance_dir, file)
        audio_input = AudioInput(file_path)
        audio, current_sr = audio_input.load_audio()
        
        if audio is not None and current_sr is not None:
            feature_extractor = PsytranceFeatureExtractor(audio, current_sr)
            features = feature_extractor.extract_features()
            processed_features = create_feature_vector(features)
            features_list.append(processed_features)
            sr = current_sr
            logging.info(f"Processed file: {file}")
        else:
            logging.warning(f"Skipping file {file} due to loading error")

    if not features_list:
        logging.error("No audio files were successfully processed. Exiting.")
        return

    features_array = np.array(features_list)
    logging.info(f"Feature array shape: {features_array.shape}")

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_array)

    features_tensor = torch.FloatTensor(normalized_features)
    
    input_dim = features_tensor.shape[1]
    latent_dim = 32
    model = PsytranceVAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(features_tensor)
        loss = vae_loss(recon_batch, features_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Generate new features
    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        new_features = model.decode(z).numpy()

    # Inverse transform the features
    new_features = scaler.inverse_transform(new_features)
    logging.info(f"Generated new features shape: {new_features.shape}")

    # Create AudioComposer with the generated features
    audio_composer = AudioComposer(new_features[0], sr)
    
    # Compose new audio
    new_audio = audio_composer.compose()
    
    # Save the generated audio
    output_file = os.path.join('data', 'output', 'generated_psytrance.wav')
    audio_composer.save_audio(new_audio, output_file)
    
    # Verify the saved file
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logging.info(f"Generated psytrance saved to: {output_file}")
        logging.info(f"Generated audio file size: {file_size} bytes")
        logging.info(f"Generated audio duration: {len(new_audio)/sr:.2f} seconds")
    else:
        logging.error("Failed to save generated audio file!")

if __name__ == "__main__":
    main()