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