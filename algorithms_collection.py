import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
# Set MNE log level to suppress LEDOIT_WOLF messages
import mne
mne.set_log_level('warning')  # Only show warnings and errors
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
# Import pyriemann for Riemannian geometry methods
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from braindecode.models import EEGNetv4, EEGInceptionMI, EEGNeX, ShallowFBCSPNet, MSVTNet, IFNet, EEGConformer, CTNet, ATCNet, EEGSimpleConv
# Try to import optional models
try:
    from braindecode.models import EEGTCNet
except ImportError:
    EEGTCNet = None
    print("Warning: EEGTCNet not available in braindecode")
try:
    from braindecode.models import SincShallowNet
except ImportError:
    SincShallowNet = None
    print("Warning: SincShallowNet not available in braindecode")
try:
    from braindecode.models import EEGITNet
except ImportError:
    EEGITNet = None
    print("Warning: EEGITNet not available in braindecode")
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from config.algorithms_config import RANDOM_STATE
from scipy import signal


class PrintLogCallback:
    def __init__(self, print_freq=50):
        self.print_freq = print_freq
        self.epoch = 0
    
    def __call__(self, net, **kwargs):
        self.epoch += 1
        if self.epoch % self.print_freq == 0 or self.epoch == 1:
            if hasattr(net, 'history_'):
                history = net.history_
                if len(history) > 0:
                    last_epoch = history[-1]
                    if 'train_loss' in last_epoch:
                        print(f"  Epoch {self.epoch}, Loss: {last_epoch['train_loss']:.4f}")
                    if 'dur' in last_epoch:
                        print(f"  Epoch {self.epoch}, Duration: {last_epoch['dur']:.2f}s")
    
    def initialize(self):
        """Initialize the callback."""
        pass
    
    def on_train_begin(self, net, **kwargs):
        """Called when training begins."""
        self.epoch = 0
    
    def on_train_end(self, net, **kwargs):
        """Called when training ends."""
        pass
    
    def on_epoch_begin(self, net, **kwargs):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, net, **kwargs):
        """Called at the end of each epoch."""
        self.__call__(net, **kwargs)
    
    def on_batch_begin(self, net, **kwargs):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, net, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_grad_computed(self, net, **kwargs):
        """Called after gradients are computed."""
        pass
    
    def set_params(self, **params):
        """Set parameters for the callback."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def apply_bandpass_filter(data, low_freq, high_freq, fs):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


class CSPLDA:
    def __init__(self, n_components=8):
        self.pipeline = Pipeline([
            ('csp', CSP(n_components=n_components, reg='ledoit_wolf', log=True)),
            ('lda', LDA())
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class CSPSVM:
    def __init__(self, n_components=8, C=1.0, kernel='rbf'):
        self.pipeline = Pipeline([
            ('csp', CSP(n_components=n_components, reg='ledoit_wolf', log=True)),
            ('svm', SVC(C=C, kernel=kernel, probability=True, random_state=RANDOM_STATE))
        ])
    
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class RiemannTangentSpace:
    """Riemannian tangent space algorithm for EEG classification.
    
    This algorithm uses Riemannian geometry to process EEG signals by:
    1. Computing covariance matrices from EEG channels
    2. Projecting covariance matrices to tangent space
    3. Using a classifier for classification
    """
    def __init__(self, estimator='oas', metric='riemann', classifier='lda', n_components=None):
        """Initialize the Riemann tangent space classifier.
        
        Args:
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for tangent space projection ('riemann', 'euclid', 'logeuclid')
            classifier: Classifier to use ('lda', 'svm', 'rf')
            n_components: Number of components for dimensionality reduction (None for no reduction)
        """
        steps = [
            ('cov', Covariances(estimator=estimator)),
            ('ts', TangentSpace(metric=metric))
        ]
        
        # Add dimensionality reduction if specified
        if n_components is not None:
            from sklearn.decomposition import PCA
            steps.append(('pca', PCA(n_components=n_components)))
        
        # Add classifier
        if classifier == 'lda':
            steps.append(('clf', LDA()))
        elif classifier == 'svm':
            from sklearn.svm import SVC
            steps.append(('clf', SVC(kernel='rbf', C=1.0, probability=True)))
        elif classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            steps.append(('clf', RandomForestClassifier(n_estimators=100, random_state=42)))
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        self.pipeline = Pipeline(steps)
        self.estimator = estimator
        self.metric = metric
        self.classifier = classifier
        self.n_components = n_components
    
    def fit(self, X, y):
        """Fit the model to the training data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        return self.pipeline.predict_proba(X)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class FilterBankTangentSpace:
    """Filter Bank Tangent Space algorithm for EEG classification.
    
    This algorithm combines multi-band filtering with Riemannian tangent space:
    1. Apply bandpass filters to multiple frequency bands
    2. Compute covariance matrices for each band
    3. Project each band's covariance matrices to tangent space
    4. Concatenate features from all bands
    5. Apply feature selection to avoid overfitting
    6. Classify using SVM or LDA
    
    Reference: Combines ideas from FBCSP and Riemannian Tangent Space
    """
    def __init__(self, n_bands=9, estimator='oas', metric='riemann', 
                 classifier='svm', n_features=100, fs=250):
        """Initialize Filter Bank Tangent Space classifier.
        
        Args:
            n_bands: Number of frequency bands (default: 9)
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for tangent space projection ('riemann', 'euclid', 'logeuclid')
            classifier: Classifier to use ('lda', 'svm', 'rf')
            n_features: Number of features to select (default: 100)
            fs: Sampling frequency (default: 250)
        """
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.feature_selection import SelectKBest, f_classif
        
        self.n_bands = n_bands
        self.estimator = estimator
        self.metric = metric
        self.fs = fs
        self.n_features = n_features
        self.classifier_name = classifier
        self.freq_bands = self._generate_freq_bands()
        
        self.cov_estimators = []
        self.ts_transformers = []
        self.feature_selector = None
        self.classifier = None
        
    def _generate_freq_bands(self):
        """Generate frequency bands from 4-40Hz."""
        bands = []
        for i in range(self.n_bands):
            low = 4 + i * 4
            high = 8 + i * 4
            bands.append((low, high))
        return bands
    
    def fit(self, X, y):
        """Fit the Filter Bank Tangent Space classifier.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.feature_selection import SelectKBest, f_classif
        
        n_samples = X.shape[0]
        
        features_list = []
        self.cov_estimators = []
        self.ts_transformers = []
        
        for low, high in self.freq_bands:
            print(f"    Processing band {low}-{high}Hz...")
            
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_estimator = Covariances(estimator=self.estimator)
            cov_matrices = cov_estimator.fit_transform(X_band)
            self.cov_estimators.append(cov_estimator)
            
            # Step 3: Project to tangent space
            ts_transformer = TangentSpace(metric=self.metric)
            ts_features = ts_transformer.fit_transform(cov_matrices, y)
            self.ts_transformers.append(ts_transformer)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        print(f"    Combined feature dimension: {X_combined.shape[1]}")
        
        # Step 5: Feature selection (CRITICAL to avoid overfitting)
        if self.n_features is not None:
            print(f"    Selecting top {self.n_features} features...")
            self.feature_selector = SelectKBest(f_classif, k=min(self.n_features, X_combined.shape[1]))
            X_selected = self.feature_selector.fit_transform(X_combined, y)
            print(f"    Selected feature dimension: {X_selected.shape[1]}")
        else:
            print(f"    No feature selection (using all {X_combined.shape[1]} features)...")
            self.feature_selector = None
            X_selected = X_combined
        
        # Step 6: Train classifier
        if self.classifier_name == 'lda':
            self.classifier = LDA()
        elif self.classifier_name == 'svm':
            self.classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)
        elif self.classifier_name == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
        
        self.classifier.fit(X_selected, y)
        self.classes_ = self.classifier.classes_
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        features_list = []
        
        for i, (low, high) in enumerate(self.freq_bands):
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_matrices = self.cov_estimators[i].transform(X_band)
            
            # Step 3: Project to tangent space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        
        # Step 5: Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined
        
        # Step 6: Predict
        return self.classifier.predict(X_selected)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        features_list = []
        
        for i, (low, high) in enumerate(self.freq_bands):
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_matrices = self.cov_estimators[i].transform(X_band)
            
            # Step 3: Project to tangent space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        
        # Step 5: Apply feature selection
        X_selected = self.feature_selector.transform(X_combined)
        
        # Step 6: Predict probabilities
        return self.classifier.predict_proba(X_selected)
    
    def extract_features(self, X):
        """Extract features from input data for visualization.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            features: Extracted features of shape (n_samples, n_features)
        """
        features_list = []
        
        for i, (low, high) in enumerate(self.freq_bands):
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_matrices = self.cov_estimators[i].transform(X_band)
            
            # Step 3: Project to tangent space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        
        # Step 5: Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined
        
        return X_selected
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class FBCSP:
    def __init__(self, n_components=4, n_bands=9, fs=250):
        self.n_components = n_components
        self.n_bands = n_bands
        self.fs = fs
        self.freq_bands = self._generate_freq_bands()
        self.csp_transformers = []
        self.classifier = LDA()


class MDM:
    """Minimum Distance to Mean classifier for Riemannian geometry.
    
    This is the classic Riemannian classifier that computes the distance
    between a test covariance matrix and the class mean covariance matrices
    in the Riemannian manifold.
    
    Reference: Barachant et al. (2012) "Multiclass brain-computer 
    interface classification by Riemannian geometry"
    """
    def __init__(self, estimator='oas', metric='riemann'):
        """Initialize MDM classifier.
        
        Args:
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for Riemannian distance ('riemann', 'euclid', 'logeuclid', 'logdet')
        """
        from pyriemann.estimation import Covariances
        from pyriemann.classification import MDM as pyriemann_MDM
        
        self.estimator = estimator
        self.metric = metric
        self.cov_estimator = Covariances(estimator=estimator)
        self.clf = pyriemann_MDM(metric=metric)
        self.classes_ = None
        self.cov_means_ = None
    
    def fit(self, X, y):
        """Fit the MDM classifier.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        self.clf.fit(cov_matrices, y)
        self.classes_ = self.clf.classes_
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        return self.clf.predict(cov_matrices)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        return self.clf.predict_proba(cov_matrices)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class FBCSP:
    def __init__(self, n_components=4, n_bands=9, fs=250):
        self.n_components = n_components
        self.n_bands = n_bands
        self.fs = fs
        self.freq_bands = self._generate_freq_bands()
        self.csp_transformers = []
        self.classifier = LDA()
    
    def _generate_freq_bands(self):
        bands = []
        for i in range(self.n_bands):
            low = 4 + i * 4
            high = 8 + i * 4
            bands.append((low, high))
        return bands
    
    def fit(self, X, y):
        features = []
        self.csp_transformers = []
        
        for low, high in self.freq_bands:
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True)
            csp_features = csp.fit_transform(X_band, y)
            features.append(csp_features)
            self.csp_transformers.append(csp)
        
        X_combined = np.hstack(features)
        self.classifier.fit(X_combined, y)
        return self
    
    def predict(self, X):
        features = []
        for i, (low, high) in enumerate(self.freq_bands):
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            csp_features = self.csp_transformers[i].transform(X_band)
            features.append(csp_features)
        
        X_combined = np.hstack(features)
        return self.classifier.predict(X_combined)
    
    def predict_proba(self, X):
        features = []
        for i, (low, high) in enumerate(self.freq_bands):
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            csp_features = self.csp_transformers[i].transform(X_band)
            features.append(csp_features)
        
        X_combined = np.hstack(features)
        return self.classifier.predict_proba(X_combined)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class EEGNet:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGNetv4(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        
        # EEGClassifier expects 4D input: (batch_size, channels, time, 1)
        X_4d = X_scaled[:, :, :, None]
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_4d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict(X_4d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict_proba(X_4d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class EEGNexClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGNeX(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGNeX expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGNeX expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGNeX expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class EEGInceptionClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGInceptionMI(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_4d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict(X_4d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict_proba(X_4d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class ShallowFBCSPNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_4d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict(X_4d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_4d = X_scaled[:, :, :, None]
        
        return self.clf.predict_proba(X_4d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class MSVTNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MSVTNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # MSVTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # MSVTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # MSVTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class IFNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BCI IV 2A dataset has sampling frequency of 250 Hz
        sfreq = 250
        
        self.model = IFNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            sfreq=sfreq
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # IFNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # IFNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # IFNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class EEGConformerClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGConformer(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGConformer expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGConformer expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGConformer expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class CTNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CTNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # CTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # CTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # CTNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class ATCNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ATCNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # ATCNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # ATCNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # ATCNet expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


def get_algorithm(algo_name, n_channels, n_times, n_classes=4):
    """
    Get algorithm instance by name
    
    Args:
        algo_name: Name of the algorithm
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes
        
    Returns:
        Algorithm instance
    """
    algo_name = algo_name.strip()
    
    if algo_name == 'CSP+LDA':
        return CSPLDA()
    elif algo_name == 'CSP+SVM':
        return CSPSVM()
    elif algo_name == 'FBCSP':
        return FBCSP()
    elif algo_name == 'FilterBankTangentSpace':
        return FilterBankTangentSpace()
    elif algo_name == 'FilterBankTangentSpace+SVM':
        return FilterBankTangentSpace(classifier='svm')
    elif algo_name == 'FilterBankTangentSpace+LDA':
        return FilterBankTangentSpace(classifier='lda')
    elif algo_name == 'FilterBankTangentSpace+RF':
        return FilterBankTangentSpace(classifier='rf')
    elif algo_name == 'MDM':
        return MDM()
    elif algo_name == 'RiemannTangentSpace':
        return RiemannTangentSpace()
    elif algo_name == 'RiemannTangentSpace+SVM':
        return RiemannTangentSpace(classifier='svm')
    elif algo_name == 'RiemannTangentSpace+RF':
        return RiemannTangentSpace(classifier='rf')
    elif algo_name == 'RiemannTangentSpace+PCA':
        return RiemannTangentSpace(n_components=10)
    elif algo_name == 'EEGNet':
        return EEGNet(n_channels, n_times, n_classes)
    elif algo_name == 'EEGNex':
        return EEGNexClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'EEG-Inception':
        return EEGInceptionClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'ShallowFBCSPNet':
        return ShallowFBCSPNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'MSVTNet':
        return MSVTNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'IFNet':
        return IFNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'EEGConformer':
        return EEGConformerClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'CTNet':
        return CTNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'ATCNet':
        return ATCNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'EEGSimpleConv':
        return EEGSimpleConvClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'EEGTCNet' and EEGTCNet is not None:
        return EEGTCNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'SincShallowNet' and SincShallowNet is not None:
        return SincShallowNetClassifier(n_channels, n_times, n_classes)
    elif algo_name == 'EEGITNet' and EEGITNet is not None:
        return EEGITNetClassifier(n_channels, n_times, n_classes)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


class EEGSimpleConvClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BCI IV 2A dataset has sampling frequency of 250 Hz
        sfreq = 250
        
        self.model = EEGSimpleConv(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            sfreq=sfreq
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGSimpleConv expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        # Only pass epochs parameter, not scheduler
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGSimpleConv expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        # EEGSimpleConv expects 3D input: (batch_size, channels, time)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class EEGTCNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        if EEGTCNet is None:
            raise ImportError("EEGTCNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGTCNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class SincShallowNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        if SincShallowNet is None:
            raise ImportError("SincShallowNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BCI IV 2A dataset has sampling frequency of 250 Hz
        sfreq = 250
        
        self.model = SincShallowNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            sfreq=sfreq
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


class EEGITNetClassifier:
    def __init__(self, n_channels, n_times, n_classes=4):
        if EEGITNet is None:
            raise ImportError("EEGITNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EEGITNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times
        )
        
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0005,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50)]
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=300, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        self.clf.fit(X_3d, y, epochs=epochs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict(X_3d)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        X_3d = X_scaled
        
        return self.clf.predict_proba(X_3d)
    
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model


def get_algorithm(algorithm_name, n_channels, n_times, n_classes=4):
    if algorithm_name == 'CSP+LDA':
        return CSPLDA(n_components=8)
    elif algorithm_name == 'CSP+SVM':
        return CSPSVM(n_components=8, C=1.0, kernel='rbf')
    elif algorithm_name == 'FBCSP':
        return FBCSP(n_components=4, n_bands=9, fs=250)
    elif algorithm_name == 'FilterBankTangentSpace':
        return FilterBankTangentSpace(n_bands=9, estimator='oas', metric='riemann', classifier='svm', n_features=100, fs=250)
    elif algorithm_name == 'FilterBankTangentSpace+SVM':
        return FilterBankTangentSpace(n_bands=9, estimator='oas', metric='riemann', classifier='svm', n_features=100, fs=250)
    elif algorithm_name == 'FilterBankTangentSpace+LDA':
        return FilterBankTangentSpace(n_bands=9, estimator='oas', metric='riemann', classifier='lda', n_features=100, fs=250)
    elif algorithm_name == 'FilterBankTangentSpace+RF':
        return FilterBankTangentSpace(n_bands=9, estimator='oas', metric='riemann', classifier='rf', n_features=100, fs=250)
    elif algorithm_name == 'MDM':
        return MDM()
    elif algorithm_name == 'RiemannTangentSpace':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='svm')
    elif algorithm_name == 'RiemannTangentSpace+SVM':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='svm')
    elif algorithm_name == 'RiemannTangentSpace+RF':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='rf')
    elif algorithm_name == 'RiemannTangentSpace+PCA':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='svm', n_components=10)
    elif algorithm_name == 'EEGNet':
        return EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEGNex':
        return EEGNexClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEG-Inception':
        return EEGInceptionClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'ShallowFBCSPNet':
        return ShallowFBCSPNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'MSVTNet':
        return MSVTNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'IFNet':
        return IFNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEGConformer':
        return EEGConformerClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'CTNet':
        return CTNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'ATCNet':
        return ATCNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEGSimpleConv':
        return EEGSimpleConvClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEGTCNet':
        if EEGTCNet is None:
            raise ImportError("EEGTCNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        return EEGTCNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'SincShallowNet':
        if SincShallowNet is None:
            raise ImportError("SincShallowNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        return SincShallowNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algorithm_name == 'EEGITNet':
        if EEGITNet is None:
            raise ImportError("EEGITNet is not available in braindecode. Please update braindecode or use a different algorithm.")
        return EEGITNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


if __name__ == '__main__':
    """Main function to print model structure when the file is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Print model structure')
    parser.add_argument('--model', type=str, default='EEGNet',
                        choices=['CSP+LDA', 'CSP+SVM', 'FBCSP', 'MDM', 'RiemannTangentSpace', 'RiemannTangentSpace+SVM', 'RiemannTangentSpace+RF', 'RiemannTangentSpace+PCA', 'EEGNet', 'EEGNex', 'EEG-Inception', 
                                 'ShallowFBCSPNet', 'MSVTNet', 'IFNet', 'EEGConformer', 'CTNet', 'ATCNet', 'EEGSimpleConv', 'EEGTCNet', 'SincShallowNet', 'EEGITNet'],
                        help='Model name to print structure')
    parser.add_argument('--channels', type=int, default=22,
                        help='Number of channels')
    parser.add_argument('--times', type=int, default=1000,
                        help='Number of time points')
    parser.add_argument('--classes', type=int, default=4,
                        help='Number of classes')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"Model Structure: {args.model}")
    print(f"{'=' * 80}")
    
    try:
        model = get_algorithm(args.model, args.channels, args.times, args.classes)
        
        if hasattr(model, 'model'):
            # For deep learning models
            print("\nDeep Learning Model Structure:")
            print(model.model)
            print(f"\nModel Parameters: {sum(p.numel() for p in model.model.parameters())}")
        else:
            # For traditional ML models
            print("\nTraditional ML Model Structure:")
            if hasattr(model, 'pipeline'):
                print(model.pipeline)
            else:
                print(model)
        
        print(f"\n{'=' * 80}")
        print("Model structure printed successfully!")
        print(f"{'=' * 80}")
    except Exception as e:
        print(f"\nError printing model structure: {e}")
        print(f"{'=' * 80}")
