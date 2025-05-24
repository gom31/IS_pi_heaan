import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """훈련된 모델 로더"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.metadata = None
        self.feature_names = None
        
    def load_all(self):
        """모든 모델 관련 파일 로드"""
        try:
            # 1. 메타데이터 로드
            self._load_metadata()
            
            # 2. 메인 모델 로드
            self._load_model()
            
            # 3. 전처리기들 로드
            self._load_preprocessors()
            
            logger.info(f"모델 로드 완료: {self.metadata['model_info']['name']}")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def _load_metadata(self):
        """메타데이터 로드"""
        metadata_path = self.model_dir / 'model_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"메타데이터 파일이 없습니다: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['data_info']['features']
    
    def _load_model(self):
        """메인 모델 로드"""
        model_path = self.model_dir / 'hiv_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def _load_preprocessors(self):
        """전처리기들 로드"""
        # 스케일러
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # 라벨 인코더들
        encoders_path = self.model_dir / 'label_encoders.pkl'
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
    
    def predict(self, features):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 전처리
        features_processed = self.preprocess_features(features)
        
        # 예측
        prediction = self.model.predict_proba(features_processed.reshape(1, -1))[0, 1]
        
        return float(prediction)
    
    def preprocess_features(self, features):
        """특성 전처리"""
        # numpy 배열로 변환
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        
        # 스케일링
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))[0]
        
        return features
    
    def get_model_info(self):
        """모델 정보 반환"""
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
    """입력 특성 검증기"""
    
    def __init__(self, feature_names, feature_mappings=None):
        self.feature_names = feature_names
        self.feature_mappings = feature_mappings or {}
        
        # 특성별 유효 범위 정의
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
        """사용자 입력 검증"""
        errors = []
        validated_features = []
        
        # 프론트엔드 필드명 -> 모델 특성명 매핑
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
            'f_t_resu': 'F_T_Resu'
        }
        
        # 모델의 특성 순서에 맞게 데이터 구성
        for feature_name in self.feature_names:
            # 역매핑으로 프론트엔드 필드명 찾기
            frontend_field = None
            for frontend_key, model_key in field_mapping.items():
                if model_key == feature_name:
                    frontend_field = frontend_key
                    break
            
            if frontend_field and frontend_field in user_inputs:
                value = user_inputs[frontend_field]
                
                # 타입 변환
                try:
                    if isinstance(value, str):
                        value = float(value)
                    value = int(value)
                except (ValueError, TypeError):
                    errors.append(f"{feature_name}: 유효한 숫자여야 합니다")
                    value = 0
                
                # 범위 검증
                if feature_name in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature_name]
                    if value < min_val or value > max_val:
                        errors.append(f"{feature_name}: {min_val}-{max_val} 범위여야 합니다")
                        value = max(min_val, min(max_val, value))  # 클리핑
                
                validated_features.append(float(value))
            else:
                validated_features.append(0.0)  # 기본값
        
        return validated_features, errors

def load_model(model_dir):
    """모델 로드 헬퍼 함수"""
    loader = ModelLoader(model_dir)
    loader.load_all()
    return loader

def validate_user_input(user_inputs, feature_names):
    """사용자 입력 검증 헬퍼 함수"""
    validator = FeatureValidator(feature_names)
    return validator.validate_features(user_inputs)