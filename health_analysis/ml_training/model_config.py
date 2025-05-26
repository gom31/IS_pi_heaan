# health_analysis/ml_training/model_config.py

"""
모델 훈련 관련 설정들
"""

# model hyper-parameter
MODEL_CONFIG = {
    'logistic_regression': {
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 2000, 3000],
            'solver': ['lbfgs', 'liblinear'],
            'class_weight': [None, 'balanced']
        },
        'cv_folds': 5,
        'scoring': 'roc_auc',
        'random_state': 42
    },
    
    'random_forest': {
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'cv_folds': 5,
        'scoring': 'roc_auc',
        'random_state': 42
    },
    
    'gradient_boosting': {
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'cv_folds': 3,
        'scoring': 'roc_auc',
        'random_state': 42
    }
}

# data preprocessing config 
DATA_CONFIG = {
    'test_size': 0.2,
    'random_state': 42
}
# model evaluation criteria
EVALUATION_CONFIG = {
    'primary_metric': 'roc_auc',
    'secondary_metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'threshold': 0.5,
    'cv_folds': 5
}

# file path config
PATHS = {
    'data_dir': '../../data',
    'model_dir': '../models',
    'log_dir': '../../logs',
    'backup_dir': '../models/backup'
}

#  HE config (CKKS parameter)
HE_CONFIG = {
    'context_params': {
        'key_dir_path': './keys',
        'generate_keys': True
    },
    'precision_bits': 40,
    'scale_bits': 20
}

# model metadata template
METADATA_TEMPLATE = {
    'model_info': {
        'name': '',
        'version': '1.0',
        'type': '',
        'framework': 'scikit-learn',
        'created_date': '',
        'description': 'HIV/AIDS risk prediction model'
    },
    'data_info': {
        'dataset': 'HIV_AIDS_DataSet.csv',
        'features': [],
        'feature_count': 0,
        'sample_count': 0,
        'target': 'F_T_Resu'
    },
    'performance': {
        'train_metrics': {},
        'test_metrics': {},
        'cross_validation': {},
        'feature_importance': {}
    },
    'model_params': {},
    'preprocessing': {
        'scaler': 'StandardScaler',
        'encoders': {},
        'feature_mappings': {}
    }
}