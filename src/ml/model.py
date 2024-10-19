import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class StyleTransfer:
    def __init__(self, n_components=None):
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components = n_components

    def fit(self, features):
        # Normalize the features
        normalized_features = self.scaler.fit_transform(features)
        
        # Determine the number of components
        n_samples, n_features = normalized_features.shape
        if self.n_components is None or self.n_components > min(n_samples, n_features):
            self.n_components = min(n_samples, n_features) - 1

        # Initialize and fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(normalized_features)

    def transform(self, original_features, style_features, alpha=0.5):
        # Ensure the features are 2D arrays
        orig_features = original_features.reshape(1, -1)
        style_features = style_features.reshape(1, -1)

        # Normalize both sets of features
        orig_norm = self.scaler.transform(orig_features)
        style_norm = self.scaler.transform(style_features)

        # Project both onto PCA space
        orig_pca = self.pca.transform(orig_norm)
        style_pca = self.pca.transform(style_norm)

        # Interpolate in PCA space
        interpolated = alpha * orig_pca + (1 - alpha) * style_pca

        # Project back to original space
        transformed = self.pca.inverse_transform(interpolated)

        # Denormalize
        return self.scaler.inverse_transform(transformed)