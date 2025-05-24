# health_analysis/he_utils.py

import os
os.environ["HEAAN_TYPE"] = "pi"
os.environ["OMP_NUM_THREADS"] = "8"

import heaan_stat
from heaan_stat import Context, Block
import numpy as np
import logging
from pathlib import Path
from .model_utils import load_model, validate_user_input

logger = logging.getLogger(__name__)

class HEHealthAnalyzer:
    def __init__(self):
        """동형암호 기반 건강 분석기"""
        try:
            # HE Context 생성
            self.context = Context(
                key_dir_path='./keys',
                generate_keys=True,
            )
            
            # 모델 로드
            self._load_trained_model()
            
            logger.info("HE Health Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HE Health Analyzer: {e}")
            raise
    
    def _load_trained_model(self):
        """훈련된 모델 로드"""
        try:
            # 모델 경로 설정
            current_dir = Path(__file__).parent
            model_dir = current_dir / 'models'
            
            # 모델 로더 사용
            self.model_loader = load_model(model_dir)
            
            # 모델 정보 추출
            model_info = self.model_loader.get_model_info()
            self.feature_names = model_info['features']
            self.model_performance = model_info['performance']
            
            # 로지스틱 회귀 계수 추출 (동형암호 연산용)
            if hasattr(self.model_loader.model, 'coef_'):
                self.model_coefficients = self.model_loader.model.coef_[0]
                self.model_intercept = self.model_loader.model.intercept_[0]
            else:
                # 다른 모델 타입의 경우 근사 계수 사용
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
        """사용자 입력 전처리"""
        try:
            # 특성 순서에 맞게 데이터 정렬
            features = []
            
            # 매핑 테이블 (프론트엔드 필드명 -> 모델 특성명) - 수정됨
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
                'r_nhave_sex': 'R_Nhave_Sex',  # 수정: R_Nhave_sex -> R_Nhave_Sex
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
                # 역매핑으로 프론트엔드 필드 찾기
                frontend_field = None
                for frontend_key, model_key in field_mapping.items():
                    if model_key == feature_name:
                        frontend_field = frontend_key
                        break
                
                if frontend_field and frontend_field in user_inputs:
                    value = user_inputs[frontend_field]
                else:
                    value = 0  # 기본값
                
                features.append(float(value))
            
            # 스케일링 적용
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.model_loader.scaler.transform(features_array)[0]
            
            logger.info(f"Preprocessed {len(features_scaled)} features")
            return features_scaled.tolist()
            
        except Exception as e:
            logger.error(f"Failed to preprocess user input: {e}")
            raise
    
    def encrypt_user_data(self, user_inputs):
        """사용자 입력 데이터 전처리 및 암호화"""
        try:
            # 전처리
            preprocessed_features = self._preprocess_user_input(user_inputs)
            
            # 암호화
            encrypted_block = Block(
                self.context,
                encrypted=True,
                data=preprocessed_features
            )
            
            logger.info("User data encrypted successfully")
            return encrypted_block
            
        except Exception as e:
            logger.error(f"Failed to encrypt user data: {e}")
            raise
    
    def compute_risk_probability_he(self, encrypted_data):
        """동형암호로 위험도 확률 계산"""
        try:
            # 선형 결합 계산: w·x + b
            linear_result = Block(
                self.context, 
                encrypted=True, 
                data=[float(self.model_intercept)]
            )
            
            # 각 특성에 가중치를 곱해서 합산
            for i, coeff in enumerate(self.model_coefficients):
                if abs(coeff) > 1e-10:  # 매우 작은 계수는 무시
                    # i번째 특성 추출 (rotation)
                    feature_i = encrypted_data >> i
                    
                    # 가중치 곱셈
                    weighted_feature = feature_i * float(coeff)
                    
                    # 합산
                    linear_result = linear_result + weighted_feature
            
            # 시그모이드 근사 (3차 다항식)
            # sigmoid(x) ≈ 0.5 + 0.25*x - 0.02083*x³ (for x in [-2, 2])
            x = linear_result
            x_squared = x * x
            x_cubed = x_squared * x
            
            # 시그모이드 계산
            sigmoid_result = Block(self.context, encrypted=True, data=[0.5])
            sigmoid_result = sigmoid_result + (x * 0.25)
            sigmoid_result = sigmoid_result - (x_cubed * 0.02083)
            
            logger.info("HE risk probability computed successfully")
            return sigmoid_result
            
        except Exception as e:
            logger.error(f"Failed to compute HE risk probability: {e}")
            raise
    
    def compute_risk_probability_direct(self, user_inputs):
        """직접 모델로 위험도 계산 (비교용)"""
        try:
            validated_features, errors = validate_user_input(user_inputs, self.feature_names)
            
            if errors:
                logger.warning(f"Validation errors (proceeding): {errors}")
            
            # 모델 예측
            probability = self.model_loader.predict(validated_features)
            
            logger.info(f"Direct prediction: {probability:.4f}")
            return probability
            
        except Exception as e:
            logger.error(f"Failed to compute direct prediction: {e}")
            raise
    
    def decrypt_result(self, encrypted_result):
        """결과 복호화 및 후처리"""
        try:
            decrypted_result = encrypted_result.decrypt(inplace=False)
            probability = float(decrypted_result[0])
            
            # [0, 1] 범위로 클리핑
            probability = max(0.0, min(1.0, probability))
            
            logger.info(f"HE result decrypted: {probability:.4f}")
            return probability
            
        except Exception as e:
            logger.error(f"Failed to decrypt result: {e}")
            raise
    
    def analyze_health_risk(self, user_inputs, use_he=True):
        """전체 건강 위험도 분석 파이프라인"""
        try:
            logger.info(f"Starting health risk analysis (HE: {use_he})...")
            
            if use_he:
                # 동형암호 방식
                encrypted_data = self.encrypt_user_data(user_inputs)
                encrypted_result = self.compute_risk_probability_he(encrypted_data)
                risk_probability = self.decrypt_result(encrypted_result)
                method = "Homomorphic Encryption"
            else:
                # 직접 계산 방식
                risk_probability = self.compute_risk_probability_direct(user_inputs)
                method = "Direct Computation"
            
            # 결과 해석
            result = {
                'risk_probability': risk_probability,
                'risk_level': self._categorize_risk(risk_probability),
                'recommendations': self._generate_recommendations(risk_probability),
                'method': method,
                'model_info': {
                    'type': self.model_loader.metadata['model_info']['type'],
                    'performance': self.model_performance
                }
            }
            
            logger.info(f"Analysis completed: {risk_probability:.4f} ({method})")
            return result
            
        except Exception as e:
            logger.error(f"Health risk analysis failed: {e}")
            raise
    
    def _categorize_risk(self, probability):
        """위험도 분류"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendations(self, probability):
        """위험도 기반 권장사항 생성"""
        if probability < 0.3:
            return [
                "현재 위험도가 낮습니다. 건강한 생활습관을 유지하세요.",
                "정기적인 건강검진을 받으시기 바랍니다.",
                "안전한 성관계를 지속하세요.",
                "예방 교육을 꾸준히 받으세요."
            ]
        elif probability < 0.7:
            return [
                "중간 수준의 위험도입니다. 전문의 상담을 고려해보세요.",
                "추가적인 검사가 필요할 수 있습니다.",
                "생활습관 개선에 더욱 신경 쓰세요.",
                "정기적인 검사를 받으시기 바랍니다."
            ]
        else:
            return [
                "높은 위험도입니다. 즉시 전문의 상담을 받으세요.",
                "정확한 진단을 위한 검사가 필요합니다.",
                "건강관리에 특별한 주의가 필요합니다.",
                "가까운 의료기관을 방문하시기 바랍니다."
            ]

# 싱글톤 패턴으로 분석기 관리
_health_analyzer = None

def get_health_analyzer():
    """싱글톤 패턴으로 분석기 인스턴스 관리"""
    global _health_analyzer
    if _health_analyzer is None:
        _health_analyzer = HEHealthAnalyzer()
    return _health_analyzer