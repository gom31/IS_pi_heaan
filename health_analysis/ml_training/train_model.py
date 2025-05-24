# health_analysis/ml_training/train_model.py 수정

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import HIVDataPreprocessor
from model_config import MODEL_CONFIG, DATA_CONFIG, EVALUATION_CONFIG, PATHS, METADATA_TEMPLATE

class HIVModelTrainer:
    def __init__(self, data_path, model_save_path):
        """HIV/AIDS 예측 모델 훈련기"""
        self.data_path = data_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 백업 디렉토리 생성
        self.backup_path = self.model_save_path / 'backup'
        self.backup_path.mkdir(exist_ok=True)
        
        self.preprocessor = HIVDataPreprocessor(data_path)
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessing_data = None
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """로지스틱 회귀 모델 훈련"""
        print("\n" + "="*50)
        print("로지스틱 회귀 모델 훈련 시작")
        print("="*50)
        
        config = MODEL_CONFIG['logistic_regression']
        
        # 그리드 서치로 최적 하이퍼파라미터 찾기
        lr = LogisticRegression(random_state=config['random_state'])
        grid_search = GridSearchCV(
            lr, config['param_grid'], 
            cv=config['cv_folds'], 
            scoring=config['scoring'], 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # 예측 및 평가
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # 교차 검증
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                  cv=config['cv_folds'], scoring=config['scoring'])
        
        result = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'coefficients': best_model.coef_[0].tolist(),
            'intercept': float(best_model.intercept_[0])
        }
        
        self._print_model_results('Logistic Regression', result)
        return result
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """랜덤 포레스트 모델 훈련"""
        print("\n" + "="*50)
        print("랜덤 포레스트 모델 훈련 시작")
        print("="*50)
        
        config = MODEL_CONFIG['random_forest']
        
        rf = RandomForestClassifier(random_state=config['random_state'])
        grid_search = GridSearchCV(
            rf, config['param_grid'],
            cv=config['cv_folds'],
            scoring=config['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        cv_scores = cross_val_score(best_model, X_train, y_train,
                                  cv=config['cv_folds'], scoring=config['scoring'])
        
        result = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': best_model.feature_importances_.tolist()
        }
        
        self._print_model_results('Random Forest', result)
        return result
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """그래디언트 부스팅 모델 훈련"""
        print("\n" + "="*50)
        print("그래디언트 부스팅 모델 훈련 시작")
        print("="*50)
        
        config = MODEL_CONFIG['gradient_boosting']
        
        gb = GradientBoostingClassifier(random_state=config['random_state'])
        grid_search = GridSearchCV(
            gb, config['param_grid'],
            cv=config['cv_folds'],
            scoring=config['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        cv_scores = cross_val_score(best_model, X_train, y_train,
                                  cv=config['cv_folds'], scoring=config['scoring'])
        
        result = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': best_model.feature_importances_.tolist()
        }
        
        self._print_model_results('Gradient Boosting', result)
        return result
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """모델 성능 지표 계산"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def _print_model_results(self, model_name, result):
        """모델 결과 출력"""
        print(f"\n{model_name} 결과:")
        print(f"최적 파라미터: {result['best_params']}")
        print(f"테스트 성능:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print(f"교차검증: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
    
    def select_best_model(self):
        """최적 모델 선택 (AUC 기준)"""
        best_auc = 0
        best_name = None
        
        print("\n" + "="*50)
        print("모델 성능 비교")
        print("="*50)
        
        for name, result in self.trained_models.items():
            auc = result['metrics']['roc_auc']  
            print(f"{name}: AUC = {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        
        print(f"\n선택된 최적 모델: {best_name} (AUC: {best_auc:.4f})")
        return self.best_model
    
    def save_model_and_artifacts(self):
        """모델과 관련 파일들 저장"""
        if not self.best_model or not self.preprocessing_data:
            raise ValueError("훈련된 모델이나 전처리 데이터가 없습니다.")
        
        print(f"\n모델 저장 중: {self.model_save_path}")
        
        # 1. 메인 모델 저장
        model_file = self.model_save_path / 'hiv_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.best_model['model'], f)
        
        # 2. 스케일러 저장
        scaler_file = self.model_save_path / 'scaler.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.preprocessing_data['scaler'], f)
        
        # 3. 라벨 인코더 저장
        encoders_file = self.model_save_path / 'label_encoders.pkl'
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.preprocessing_data['label_encoders'], f)
        
        # 4. 메타데이터 생성 및 저장
        metadata = self._create_metadata()
        metadata_file = self.model_save_path / 'model_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 5. 백업 저장
        self._save_backup()
        
        print("✅ 모델 저장 완료!")
        return metadata
    
    def _create_metadata(self):
        """모델 메타데이터 생성"""
        metadata = METADATA_TEMPLATE.copy()
        
        # 모델 정보
        metadata['model_info'].update({
            'name': f'HIV_Risk_Predictor_{self.best_model_name}',
            'type': self.best_model_name,
            'created_date': datetime.now().isoformat(),
        })
        
        # 데이터 정보
        metadata['data_info'].update({
            'features': self.preprocessing_data['features'],
            'feature_count': len(self.preprocessing_data['features']),
            'sample_count': len(self.preprocessing_data['X_train']) + len(self.preprocessing_data['X_test'])
        })
        
        # 성능 정보
        metadata['performance'].update({
            'test_metrics': self.best_model['metrics'],
            'cross_validation': {
                'mean_auc': self.best_model['cv_mean'],
                'std_auc': self.best_model['cv_std']
            },
            'best_params': self.best_model['best_params']
        })
        
        # 전처리 정보
        metadata['preprocessing'].update({
            'feature_mappings': self.preprocessing_data.get('feature_mappings', {})
        })
        
        # 모델별 특별 정보
        if 'coefficients' in self.best_model:
            metadata['model_params']['coefficients'] = self.best_model['coefficients']
            metadata['model_params']['intercept'] = self.best_model['intercept']
        
        if 'feature_importance' in self.best_model:
            # 특성 중요도와 특성명 매핑
            feature_importance = dict(zip(
                self.preprocessing_data['features'],
                self.best_model['feature_importance']
            ))
            metadata['performance']['feature_importance'] = feature_importance
        
        return metadata
    
    def _save_backup(self):
        """백업 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_path / f'hiv_model_{self.best_model_name}_{timestamp}.pkl'
        
        with open(backup_file, 'wb') as f:
            pickle.dump({
                'model': self.best_model['model'],
                'metadata': self._create_metadata(),
                'preprocessing': self.preprocessing_data
            }, f)
    
    def train_all_models(self):
        """모든 모델 훈련 파이프라인"""
        print("="*80)
        print("HIV/AIDS 위험도 예측 모델 훈련 시작")
        print("="*80)
        
        # 1. 데이터 전처리 (수정: DATA_CONFIG 대신 직접 매개변수 전달)
        print("\n1. 데이터 전처리 중...")
        self.preprocessing_data = self.preprocessor.preprocess(
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state']
        )
        
        X_train = self.preprocessing_data['X_train']
        X_test = self.preprocessing_data['X_test']
        y_train = self.preprocessing_data['y_train']
        y_test = self.preprocessing_data['y_test']
        
        # 2. 모델들 훈련
        print("\n2. 모델 훈련 중...")
        
        # 로지스틱 회귀
        self.trained_models['Logistic_Regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        
        # 랜덤 포레스트
        self.trained_models['Random_Forest'] = self.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # 그래디언트 부스팅
        self.trained_models['Gradient_Boosting'] = self.train_gradient_boosting(
            X_train, y_train, X_test, y_test
        )
        
        # 3. 최적 모델 선택
        print("\n3. 최적 모델 선택 중...")
        self.select_best_model()
        
        # 4. 모델 저장
        print("\n4. 모델 저장 중...")
        metadata = self.save_model_and_artifacts()
        
        print("\n" + "="*80)
        print("모델 훈련 완료!")
        print("="*80)
        
        return metadata

def main():
    """메인 실행 함수"""
    # 경로 설정
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    data_path = project_root / 'data' / 'HIV_AIDS_DataSet.csv'
    model_save_path = current_dir.parent / 'models'
    
    print(f"데이터 경로: {data_path}")
    print(f"모델 저장 경로: {model_save_path}")
    
    # 경로 존재 확인
    if not data_path.exists():
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        print("HIV_AIDS_DataSet.csv 파일을 data/ 폴더에 추가해주세요.")
        print("\n다운로드 링크 예시:")
        print("- Kaggle HIV/AIDS dataset")
        print("- DHS (Demographic and Health Surveys) data")
        return
    
    # 모델 훈련 실행
    try:
        trainer = HIVModelTrainer(str(data_path), str(model_save_path))
        metadata = trainer.train_all_models()
        
        print(f"\n✅ 훈련 완료!")
        print(f"선택된 모델: {trainer.best_model_name}")
        print(f"AUC 점수: {metadata['performance']['test_metrics']['roc_auc']:.4f}")
        print(f"정확도: {metadata['performance']['test_metrics']['accuracy']:.4f}")
        
        # 파일 생성 확인
        print(f"\n생성된 파일들:")
        for file_path in model_save_path.glob("*.pkl"):
            print(f"  ✅ {file_path.name}")
        for file_path in model_save_path.glob("*.json"):
            print(f"  ✅ {file_path.name}")
            
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()