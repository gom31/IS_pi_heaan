import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Trained model loader"""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.metadata = None
        self.feature_names = None

    def load_all(self):
        """Load all model-related files"""
        try:
            # 1. Load metadata
            self._load_metadata()

            # 2. Load main model
            self._load_model()

            # 3. Load preprocessors
            self._load_preprocessors()

            logger.info(f"Model loaded: {self.metadata['model_info']['name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_metadata(self):
        """Load metadata"""
        metadata_path = self.model_dir / 'model_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['data_info']['features']

    def _load_model(self):
        """Load main model"""
        model_path = self.model_dir / 'hiv_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def _load_preprocessors(self):
        """Load preprocessors"""
        # Scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Label encoders
        encoders_path = self.model_dir / 'label_encoders.pkl'
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)

    def predict(self, features):
        """Perform prediction"""
        if self.model is None:
            raise ValueError("Model has not been loaded.")

        # Preprocessing
        features_processed = self.preprocess_features(features)

        # Prediction
        prediction = self.model.predict_proba(features_processed.reshape(1, -1))[0, 1]

        return float(prediction)

    def preprocess_features(self, features):
        """Preprocess input features"""
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)

        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        return features

    def get_model_info(self):
        """Return model information"""
        if self.metadata is None:
            return None

        return {
            'name': self.metadata['model_info']['name'],
            'type': self.metadata['model_info']['type'],
            'version': self.metadata['model_info']['version'],
            'created_date': self.metadata['model_info']['created_date'],
            'features': self.feature_names,
            'performance': self.metadata['performance']['test_metrics']
        }

class FeatureValidator:
    """Validator for input features"""

    def __init__(self, feature_names, feature_mappings=None):
        self.feature_names = feature_names
        self.feature_mappings = feature_mappings or {}

        # Valid range for each feature
        self.feature_ranges = {
            'Sex': [0, 1],
            'Age': [1, 7],
            'Edu_lvl': [0, 4],
            'Had_Sex': [0, 1],
            'N_S_Part': [0, 100],
            'Con_Use': [0, 1],
            'R_Use_Con': [0, 1],
            'R_SeA': [0, 1],
            'R_Have_1SP': [0, 1],
            'R_Nhave_Sex': [0, 1],
            'HIV_Mosq': [0, 1],
            'H_STI': [0, 1],
            'H_O_STI': [0, 1],
            'E_T_HIV': [0, 1],
            'P_T_HIV': [0, 1],
            'S_Test': [0, 1],
            'T_in_LAB': [0, 1],
            'F_T_Resu': [0, 1]
        }

    def validate_features(self, user_inputs):
        """Validate user inputs"""
        errors = []
        validated_features = []

        # Mapping from frontend field names to model feature names
        field_mapping = {
            'sex': 'Sex',
            'age': 'Age', 
            'edu_lvl': 'Edu_lvl',
            'had_sex': 'Had_Sex',
            'n_s_part': 'N_S_Part',
            'con_use': 'Con_Use',
            'r_use_con': 'R_Use_Con',
            'r_sea': 'R_SeA',
            'r_have_1sp': 'R_Have_1SP',
            'r_nhave_sex': 'R_Nhave_sex',
            'hiv_mosq': 'HIV_Mosq',
            'h_sti': 'H_STI',
            'h_o_sti': 'H_O_STI',
            'e_t_hiv': 'E_T_HIV',
            'p_t_hiv': 'P_T_HIV',
            's_test': 'S_Test',
            't_in_lab': 'T_in_LAB',
            'h_aids': 'H_AIDS'
        }

        # Build data in the order of model features
        for feature_name in self.feature_names:
            frontend_field = None
            for frontend_key, model_key in field_mapping.items():
                if model_key == feature_name:
                    frontend_field = frontend_key
                    break

            if frontend_field and frontend_field in user_inputs:
                value = user_inputs[frontend_field]

                # Type conversion
                try:
                    if isinstance(value, str):
                        value = float(value)
                    value = int(value)
                except (ValueError, TypeError):
                    errors.append(f"{feature_name}: must be a valid number")
                    value = 0

                # Range validation
                if feature_name in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature_name]
                    if value < min_val or value > max_val:
                        errors.append(f"{feature_name}: must be in range {min_val}-{max_val}")
                        value = max(min_val, min(max_val, value))

                validated_features.append(float(value))
            else:
                validated_features.append(0.0)  # Default value

        return validated_features, errors

def load_model(model_dir):
    """Helper function to load model"""
    loader = ModelLoader(model_dir)
    loader.load_all()
    return loader

def validate_user_input(user_inputs, feature_names):
    """Helper function to validate user input"""
    validator = FeatureValidator(feature_names)
    return validator.validate_features(user_inputs)
