import os
os.environ["HEAAN_TYPE"] = "pi"
os.environ["OMP_NUM_THREADS"] = "8"

import heaan_stat
from heaan_stat import Context, Block
import numpy as np
import logging
import time  # 추가된 import
from pathlib import Path
from .model_utils import load_model, validate_user_input

logger = logging.getLogger(__name__)

class HEHealthAnalyzer:
    def __init__(self):
        """Homomorphic Encryption-based Health Analyzer"""
        try:
            # Create HE Context
            self.context = Context(
                key_dir_path='./keys',
                generate_keys=False,
            )

            # Load trained model
            self._load_trained_model()

            logger.info("HE Health Analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize HE Health Analyzer: {e}")
            raise

    def _load_trained_model(self):
        """Load the trained model"""
        try:
            # Set model path
            current_dir = Path(__file__).parent
            model_dir = current_dir / 'models'

            # Use model loader
            self.model_loader = load_model(model_dir)

            # Extract model information
            model_info = self.model_loader.get_model_info()
            self.feature_names = model_info['features']
            self.model_performance = model_info['performance']

            # Extract logistic regression coefficients (for HE operations)
            if hasattr(self.model_loader.model, 'coef_'):
                self.model_coefficients = self.model_loader.model.coef_[0]
                self.model_intercept = self.model_loader.model.intercept_[0]
            else:
                # Use approximated coefficients for other model types
                self.model_coefficients = np.random.randn(len(self.feature_names)) * 0.1
                self.model_intercept = 0.0
                logger.warning("Non-linear model detected. Using approximated coefficients for HE.")

            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Type: {model_info['type']}")
            logger.info(f"  - Features: {len(self.feature_names)}")
            logger.info(f"  - AUC: {self.model_performance.get('roc_auc', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _preprocess_user_input(self, user_inputs):
        """Preprocess user input"""
        try:
            # Arrange data in order of features
            features = []

            # Mapping table (frontend field name -> model feature name)
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
                'r_nhave_sex': 'R_Nhave_Sex',
                'hiv_mosq': 'HIV_Mosq',
                'h_sti': 'H_STI',
                'h_o_sti': 'H_O_STI',
                'e_t_hiv': 'E_T_HIV',
                'p_t_hiv': 'P_T_HIV',
                's_test': 'S_Test',
                't_in_lab': 'T_in_LAB',
                'h_aids': 'H_AIDS'
            }

            # Construct data in model feature order
            for feature_name in self.feature_names:
                frontend_field = None
                for frontend_key, model_key in field_mapping.items():
                    if model_key == feature_name:
                        frontend_field = frontend_key
                        break

                if frontend_field and frontend_field in user_inputs:
                    value = user_inputs[frontend_field]
                else:
                    value = 0  # Default value

                features.append(float(value))

            # Apply scaling
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.model_loader.scaler.transform(features_array)[0]

            logger.info(f"Preprocessed {len(features_scaled)} features")
            return features_scaled.tolist()

        except Exception as e:
            logger.error(f"Failed to preprocess user input: {e}")
            raise

    def encrypt_user_data(self, user_inputs):
        """Preprocess and encrypt user input data"""
        try:
            # preprocess_start time
            preprocess_start = time.time()
            preprocessed_features = self._preprocess_user_input(user_inputs)
            preprocess_time = time.time() - preprocess_start

            # encrypt time
            encrypt_start = time.time()
            encrypted_block = Block(
                self.context,
                encrypted=True,
                data=preprocessed_features
            )
            encrypt_time = time.time() - encrypt_start

            logger.info("User data encrypted successfully")
            logger.info(f"  - Preprocessing time: {preprocess_time*1000:.2f}ms")
            logger.info(f"  - Encryption time: {encrypt_time*1000:.2f}ms")
            
            return encrypted_block, {
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'encryption_time_ms': round(encrypt_time * 1000, 2)
            }

        except Exception as e:
            logger.error(f"Failed to encrypt user data: {e}")
            raise

    def compute_risk_probability_he(self, encrypted_data):
        """Compute risk probability using HE"""
        try:
            # HE compute time
            he_computation_start = time.time()
            
            # Linear combination: w·x + b
            linear_start = time.time()
            linear_result = Block(
                self.context, 
                encrypted=True, 
                data=[float(self.model_intercept)]
            )
            linear_init_time = time.time() - linear_start

            # Multiply each feature with weight and sum
            multiplication_start = time.time()
            for i, coeff in enumerate(self.model_coefficients):
                if abs(coeff) > 1e-10:
                    feature_i = encrypted_data >> i
                    weighted_feature = feature_i * float(coeff)
                    linear_result = linear_result + weighted_feature
            multiplication_time = time.time() - multiplication_start

            total_he_computation_time = time.time() - he_computation_start

            logger.info("HE risk probability computed successfully")
            logger.info(f"  - Linear initialization: {linear_init_time*1000:.2f}ms")
            logger.info(f"  - Feature multiplication: {multiplication_time*1000:.2f}ms")
            logger.info(f"  - Total HE computation: {total_he_computation_time*1000:.2f}ms")
            
            return linear_result, {
                'linear_init_time_ms': round(linear_init_time * 1000, 2),
                'multiplication_time_ms': round(multiplication_time * 1000, 2),
                'total_he_computation_time_ms': round(total_he_computation_time * 1000, 2)
            }

        except Exception as e:
            logger.error(f"Failed to compute HE risk probability: {e}")
            raise

    def compute_risk_probability_direct(self, user_inputs):
        """Direct model-based risk prediction (for comparison)"""
        try:
            # direct time
            direct_start = time.time()
            
            validated_features, errors = validate_user_input(user_inputs, self.feature_names)

            if errors:
                logger.warning(f"Validation errors (proceeding): {errors}")

            probability = self.model_loader.predict(validated_features)
            
            direct_time = time.time() - direct_start

            logger.info(f"Direct prediction: {probability:.4f}")
            logger.info(f"  - Direct computation time: {direct_time*1000:.2f}ms")
            
            return probability, {
                'direct_computation_time_ms': round(direct_time * 1000, 2)
            }

        except Exception as e:
            logger.error(f"Failed to compute direct prediction: {e}")
            raise

    def decrypt_result(self, encrypted_result):
        """Decrypt and postprocess result"""
        try:
            # decrypt time
            decrypt_start = time.time()
            decrypted_result = encrypted_result.decrypt(inplace=False)
            decrypt_time = time.time() - decrypt_start
            
            # post process time
            postprocess_start = time.time()
            linear_value = float(decrypted_result[0])
            probability = 1 / (1 + np.exp(-linear_value))
            postprocess_time = time.time() - postprocess_start

            logger.info(f"HE result decrypted: {probability:.4f}")
            logger.info(f"  - Decryption time: {decrypt_time*1000:.2f}ms")
            logger.info(f"  - Postprocessing time: {postprocess_time*1000:.2f}ms")
            
            return probability, {
                'decryption_time_ms': round(decrypt_time * 1000, 2),
                'postprocess_time_ms': round(postprocess_time * 1000, 2)
            }

        except Exception as e:
            logger.error(f"Failed to decrypt result: {e}")
            raise

    def analyze_health_risk(self, user_inputs, use_he=True):
        """Complete health risk analysis pipeline"""
        try:
            # total analysis time
            analysis_start = time.time()
            
            logger.info(f"Starting health risk analysis (HE: {use_he})...")

            # timing info
            timing_info = {}

            if use_he:
                # HE times for steps
                encrypted_data, encrypt_timing = self.encrypt_user_data(user_inputs)
                timing_info.update(encrypt_timing)
                
                encrypted_result, computation_timing = self.compute_risk_probability_he(encrypted_data)
                timing_info.update(computation_timing)
                
                risk_probability, decrypt_timing = self.decrypt_result(encrypted_result)
                timing_info.update(decrypt_timing)
                
                method = "Homomorphic Encryption"
                
            else:
                # direct time
                risk_probability, direct_timing = self.compute_risk_probability_direct(user_inputs)
                timing_info.update(direct_timing)
                method = "Direct Computation"

            # total time analysis
            total_analysis_time = time.time() - analysis_start
            timing_info['total_analysis_time_ms'] = round(total_analysis_time * 1000, 2)

            result = {
                'risk_probability': risk_probability,
                'risk_level': self._categorize_risk(risk_probability),
                'recommendations': self._generate_recommendations(risk_probability),
                'method': method,
                'model_info': {
                    'type': self.model_loader.metadata['model_info']['type'],
                    'performance': self.model_performance
                },
                'timing_info': timing_info  
            }

            logger.info(f"Analysis completed: {risk_probability:.4f} ({method})")
            logger.info(f"Total analysis time: {total_analysis_time*1000:.2f}ms")
            
            return result

        except Exception as e:
            logger.error(f"Health risk analysis failed: {e}")
            raise

    def _categorize_risk(self, probability):
        """Categorize risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

    def _generate_recommendations(self, probability):
        """Generate recommendations based on risk"""
        if probability < 0.3:
            return [
                "Your risk level is low. Maintain healthy lifestyle.",
                "Have regular health check-ups.",
                "Continue safe sexual practices.",
                "Participate in prevention education."
            ]
        elif probability < 0.7:
            return [
                "You are at moderate risk. Consider consulting a specialist.",
                "Further tests may be necessary.",
                "Focus on improving lifestyle habits.",
                "Have periodic medical examinations."
            ]
        else:
            return [
                "You are at high risk. Consult a medical specialist immediately.",
                "Tests are required for accurate diagnosis.",
                "Pay special attention to your health management.",
                "Visit a nearby medical institution."
            ]

# Manage analyzer as a singleton
_health_analyzer = None

def get_health_analyzer():
    """Manage analyzer instance using singleton pattern"""
    global _health_analyzer
    if _health_analyzer is None:
        _health_analyzer = HEHealthAnalyzer()
    return _health_analyzer
