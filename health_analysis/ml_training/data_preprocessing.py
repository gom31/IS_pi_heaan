# health_analysis/ml_training/data_preprocessing.py 수정

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
import pickle
from pathlib import Path

class HIVDataPreprocessor:
    def __init__(self, data_path):
        """HIV/AIDS 데이터셋 전처리기"""
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # 19개 선택된 특성 (실제 CSV 컬럼명에 맞게 수정)
        self.selected_features = [
            'Sex', 'Age', 'Edu_lvl', 'Had_Sex', 'N_S_Part', 
            'Con_Use', 'R_Use_Con', 'R_SeA', 'R_Have_1SP', 
            'R_Nhave_Sex',  # 수정: R_Nhave_sex -> R_Nhave_Sex
            'HIV_Mosq', 'H_STI', 'H_O_STI', 
            'E_T_HIV', 'P_T_HIV', 'S_Test', 'T_in_LAB', 'F_T_Resu'
        ]
        
        # 타겟 변수
        self.target_column = 'H_AIDS'
        
        # 특성별 매핑 정의
        self.feature_mappings = {
            'Sex': {0: 'Female', 1: 'Male'},
            'Age': {1: '15-19', 2: '20-24', 3: '25-29', 4: '30-34', 
                   5: '35-39', 6: '40-44', 7: '45-49'},
            'Edu_lvl': {0: 'No education', 1: 'Primary', 2: 'Secondary', 
                       3: 'Higher', 4: 'Unknown'},
            'Had_Sex': {0: 'No', 1: 'Yes'},
            'Con_Use': {0: 'No', 1: 'Yes'},
            'R_Use_Con': {0: 'No', 1: 'Yes'},
            'R_SeA': {0: 'No', 1: 'Yes'},
            'R_Have_1SP': {0: 'No', 1: 'Yes'},
            'R_Nhave_Sex': {0: 'No', 1: 'Yes'},
            'HIV_Mosq': {0: 'No', 1: 'Yes'},
            'H_STI': {0: 'No', 1: 'Yes'},
            'H_O_STI': {0: 'No', 1: 'Yes'},
            'E_T_HIV': {0: 'No', 1: 'Yes'},
            'P_T_HIV': {0: 'No', 1: 'Yes'},
            'S_Test': {0: 'Negative', 1: 'Positive'},
            'T_in_LAB': {0: 'No', 1: 'Yes'},
            'F_T_Resu': {0: 'Negative', 1: 'Positive'}
        }
    
    def load_data(self):
        """데이터 로드 및 기본 검증"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✓ 데이터 로드 완료: {len(df)} 행, {len(df.columns)} 열")
            
            # 실제 컬럼 확인
            print(f"실제 컬럼들: {sorted(df.columns.tolist())}")
            
            # 필수 컬럼 존재 확인
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                print(f"누락된 특성들: {missing_features}")
                print("사용 가능한 컬럼들:")
                for col in sorted(df.columns):
                    if any(col.lower().replace('_', '') == f.lower().replace('_', '') for f in missing_features):
                        print(f"  비슷한 컬럼: {col}")
                raise ValueError(f"누락된 특성들: {missing_features}")
            
            if self.target_column not in df.columns:
                raise ValueError(f"타겟 컬럼 '{self.target_column}'이 없습니다.")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
    
    def clean_data(self, df):
        """데이터 정리 및 결측값 처리"""
        # 필요한 컬럼만 선택
        df_clean = df[self.selected_features + [self.target_column]].copy()
        
        # 데이터 타입 확인
        print("컬럼별 데이터 타입:")
        for col in df_clean.columns:
            print(f"  {col}: {df_clean[col].dtype} (unique: {df_clean[col].nunique()})")
        
        # 결측값 처리
        print("\n결측값 처리 전:")
        missing_counts = df_clean.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} ({count/len(df_clean)*100:.1f}%)")
        
        for col in self.selected_features:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        df_clean[self.target_column] = pd.to_numeric(df_clean[self.target_column], errors='coerce').fillna(0)
        
        # 이상값 처리
        df_clean = self._handle_outliers(df_clean)
        
        print(f"✓ 데이터 정리 완료: {len(df_clean)} 행")
        return df_clean
    
    def _handle_outliers(self, df):
        """이상값 처리"""
        # N_S_Part (성관계 파트너 수) 이상값 처리
        if 'N_S_Part' in df.columns:
            original_max = df['N_S_Part'].max()
            df.loc[df['N_S_Part'] > 50, 'N_S_Part'] = 50  # 상한선 설정
            outliers_count = (df['N_S_Part'] == 50).sum()
            if outliers_count > 0:
                print(f"  N_S_Part 이상값 처리: {outliers_count}개 (최대값: {original_max} -> 50)")
        
        return df
    
    def encode_features(self, df):
        """특성 인코딩"""
        df_encoded = df.copy()
        
        # 범주형 특성들에 대해 라벨 인코딩 (필요시)
        categorical_features = ['Sex', 'Age', 'Edu_lvl']
        
        for feature in categorical_features:
            if feature in self.selected_features:
                # 이미 정수형이면 인코딩 스킵
                if df_encoded[feature].dtype in ['int64', 'float64']:
                    continue
                    
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df_encoded
    
    def split_features_target(self, df):
        """특성과 타겟 분리"""
        X = df[self.selected_features].values.astype(np.float32)
        y = df[self.target_column].values.astype(np.int32)
        
        return X, y
    
    def preprocess(self, test_size=0.2, random_state=42):
        """전체 전처리 파이프라인"""
        print("=" * 50)
        print("HIV/AIDS 데이터 전처리 시작")
        print("=" * 50)
        
        # 1. 데이터 로드
        df = self.load_data()
        
        # 2. 데이터 정리
        df_clean = self.clean_data(df)
        
        # 3. 특성 인코딩
        df_encoded = self.encode_features(df_clean)
        
        # 4. 특성-타겟 분리
        X, y = self.split_features_target(df_encoded)
        
        # 5. 데이터 분포 확인
        self._print_data_info(X, y)
        
        # 6. 훈련/테스트 분할 (stratify 적용)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # 7. 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ 전처리 완료:")
        print(f"  - 훈련 세트: {X_train_scaled.shape}")
        print(f"  - 테스트 세트: {X_test_scaled.shape}")
        print(f"  - 양성 비율: {np.mean(y):.1%}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled, 
            'y_train': y_train,
            'y_test': y_test,
            'features': self.selected_features,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_mappings': self.feature_mappings
        }
    
    def _print_data_info(self, X, y):
        """데이터 정보 출력"""
        print(f"\n데이터 정보:")
        print(f"  - 총 샘플 수: {len(X):,}")
        print(f"  - 특성 수: {X.shape[1]}")
        print(f"  - 양성 샘플: {np.sum(y):,} ({np.mean(y):.1%})")
        print(f"  - 음성 샘플: {len(y) - np.sum(y):,} ({1-np.mean(y):.1%})")
        
        # 특성별 기본 통계
        print(f"\n특성별 기본 통계:")
        for i, feature in enumerate(self.selected_features):
            values = X[:, i]
            print(f"  {feature}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

if __name__ == "__main__":
    # 테스트 실행
    data_path = "../../data/HIV_AIDS_DataSet.csv"
    if os.path.exists(data_path):
        preprocessor = HIVDataPreprocessor(data_path)
        result = preprocessor.preprocess()
        print("전처리 테스트 완료!")
    else:
        print(f"테스트용 데이터 파일이 없습니다: {data_path}")