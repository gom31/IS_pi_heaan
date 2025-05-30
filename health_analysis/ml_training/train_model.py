import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
        """HIV/AIDS prediction model trainer"""
        self.data_path = data_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Create backup directory
        self.backup_path = self.model_save_path / 'backup'
        self.backup_path.mkdir(exist_ok=True)

        self.preprocessor = HIVDataPreprocessor(data_path)
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessing_data = None

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train logistic regression model"""
        print("\n" + "="*50)
        print("Logistic regression model training started")
        print("="*50)

        config = MODEL_CONFIG['logistic_regression']

        # Finding optimal parameters using grid search
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

        # Prediction and evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Cross-validation
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

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate model performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def _print_model_results(self, model_name, result):
        """Print model results"""
        print(f"\n{model_name} Results:")
        print(f"Best parameters: {result['best_params']}")
        print(f"Test performance:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print(f"Cross-validation: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")

    def select_best_model(self):
        """Select the best model (only logistic regression)"""
        self.best_model_name = 'Logistic_Regression'
        self.best_model = self.trained_models['Logistic_Regression']
        print(f"\nSelected best model: {self.best_model_name} (AUC: {self.best_model['metrics']['roc_auc']:.4f})")
        return self.best_model

    def save_model_and_artifacts(self):
        """Save model and related artifacts"""
        if not self.best_model or not self.preprocessing_data:
            raise ValueError("There is no trained model or preprocessing data available.")

        print(f"\nSaving models to: {self.model_save_path}")

        # 1. Save main model
        model_file = self.model_save_path / 'hiv_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.best_model['model'], f)

        # 2. Save scaler
        scaler_file = self.model_save_path / 'scaler.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.preprocessing_data['scaler'], f)

        # 3. Save label encoders
        encoders_file = self.model_save_path / 'label_encoders.pkl'
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.preprocessing_data['label_encoders'], f)

        # 4. Create and save metadata
        metadata = self._create_metadata()
        metadata_file = self.model_save_path / 'model_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 5. Save backup
        self._save_backup()

        print("✅ Model save complete!")
        return metadata

    def _create_metadata(self):
        """Create model metadata"""
        metadata = METADATA_TEMPLATE.copy()

        # Model info
        metadata['model_info'].update({
            'name': f'HIV_Risk_Predictor_{self.best_model_name}',
            'type': self.best_model_name,
            'created_date': datetime.now().isoformat(),
        })

        # Data info
        metadata['data_info'].update({
            'features': self.preprocessing_data['features'],
            'feature_count': len(self.preprocessing_data['features']),
            'sample_count': len(self.preprocessing_data['X_train']) + len(self.preprocessing_data['X_test'])
        })

        # Performance info
        metadata['performance'].update({
            'test_metrics': self.best_model['metrics'],
            'cross_validation': {
                'mean_auc': self.best_model['cv_mean'],
                'std_auc': self.best_model['cv_std']
            },
            'best_params': self.best_model['best_params']
        })

        # Preprocessing info
        metadata['preprocessing'].update({
            'feature_mappings': self.preprocessing_data.get('feature_mappings', {})
        })

        # Model parameter info
        if 'coefficients' in self.best_model:
            metadata['model_params']['coefficients'] = self.best_model['coefficients']
            metadata['model_params']['intercept'] = self.best_model['intercept']

        if 'feature_importance' in self.best_model:
            feature_importance = dict(zip(
                self.preprocessing_data['features'],
                self.best_model['feature_importance']
            ))
            metadata['performance']['feature_importance'] = feature_importance

        return metadata

    def _save_backup(self):
        """Save model backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_path / f'hiv_model_{self.best_model_name}_{timestamp}.pkl'

        with open(backup_file, 'wb') as f:
            pickle.dump({
                'model': self.best_model['model'],
                'metadata': self._create_metadata(),
                'preprocessing': self.preprocessing_data
            }, f)

    def train_all_models(self):
        """Training pipeline for logistic regression model only"""
        print("="*80)
        print("Starting HIV/AIDS Risk Prediction Model Training (Logistic Regression Only)")
        print("="*80)

        # 1. Data preprocessing
        print("\n1. Preprocessing data...")
        self.preprocessing_data = self.preprocessor.preprocess(
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state']
        )

        X_train = self.preprocessing_data['X_train']
        X_test = self.preprocessing_data['X_test']
        y_train = self.preprocessing_data['y_train']
        y_test = self.preprocessing_data['y_test']

        # 2. Train logistic regression only
        print("\n2. Training Logistic Regression model...")
        self.trained_models['Logistic_Regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )

        # 3. Select best model (logistic only)
        print("\n3. Selecting best model...")
        self.best_model_name = 'Logistic_Regression'
        self.best_model = self.trained_models['Logistic_Regression']
        print(f"\nSelected best model: {self.best_model_name} (AUC: {self.best_model['metrics']['roc_auc']:.4f})")

        # 4. Save model
        print("\n4. Saving model...")
        return self.save_model_and_artifacts()

def main():
    """Main execution function"""
    # Set paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    data_path = project_root / 'data' / 'HIV_AIDS_DataSet.csv'
    model_save_path = current_dir.parent / 'models'

    print(f"Data path: {data_path}")
    print(f"Model save path: {model_save_path}")

    # Check path existence
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please add 'HIV_AIDS_DataSet.csv' to the data/ folder.")
        print("\nExample download links:")
        print("- Kaggle HIV/AIDS dataset")
        print("- DHS (Demographic and Health Surveys) data")
        return

    # Run model training
    try:
        trainer = HIVModelTrainer(str(data_path), str(model_save_path))
        metadata = trainer.train_all_models()

        print(f"\n✅ Training complete!")
        print(f"Selected model: {trainer.best_model_name}")
        print(f"AUC score: {metadata['performance']['test_metrics']['roc_auc']:.4f}")
        print(f"Accuracy: {metadata['performance']['test_metrics']['accuracy']:.4f}")

        # Check generated files
        print(f"\nGenerated files:")
        for file_path in model_save_path.glob("*.pkl"):
            print(f"  ✅ {file_path.name}")
        for file_path in model_save_path.glob("*.json"):
            print(f"  ✅ {file_path.name}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
