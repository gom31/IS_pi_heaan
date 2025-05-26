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
        """Preprocessor for the HIV/AIDS dataset"""
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # 19 selected features (adjust to match actual CSV column names)
        self.selected_features = [
            'Sex', 'Age', 'Edu_lvl', 'Had_Sex', 'N_S_Part', 
            'Con_Use', 'R_Use_Con', 'R_SeA', 'R_Have_1SP', 
            'R_Nhave_Sex',  
            'HIV_Mosq', 'H_STI', 'H_O_STI', 
            'E_T_HIV', 'P_T_HIV', 'S_Test', 'T_in_LAB', 'H_AIDS'
        ]
        
        # Target variable
        self.target_column = 'F_T_Resu'
        
        # Feature mappings
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
            'H_AIDS': {0: 'Negative', 1: 'Positive'}
        }
    
    def load_data(self):
        """Load data and basic validation"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Check actual columns
            print(f"Available columns: {sorted(df.columns.tolist())}")
            
            # Check for required columns
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                print(f"Missing features: {missing_features}")
                print("Available similar columns:")
                for col in sorted(df.columns):
                    if any(col.lower().replace('_', '') == f.lower().replace('_', '') for f in missing_features):
                        print(f"  Similar column: {col}")
                raise ValueError(f"Missing features: {missing_features}")
            
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' is missing.")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def clean_data(self, df):
        """Clean data and handle missing values"""
        # Select only necessary columns
        df_clean = df[self.selected_features + [self.target_column]].copy()
        
        # Check data types
        print("Column data types:")
        for col in df_clean.columns:
            print(f"  {col}: {df_clean[col].dtype} (unique: {df_clean[col].nunique()})")
        
        # Handle missing values
        print("\nMissing values before handling:")
        missing_counts = df_clean.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} ({count/len(df_clean)*100:.1f}%)")
        
        for col in self.selected_features:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        df_clean[self.target_column] = pd.to_numeric(df_clean[self.target_column], errors='coerce').fillna(0)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        print(f"✓ Data cleaned: {len(df_clean)} rows")
        return df_clean
    
    def _handle_outliers(self, df):
        """Handle outliers"""
        # Handle outliers in N_S_Part (number of sexual partners)
        if 'N_S_Part' in df.columns:
            original_max = df['N_S_Part'].max()
            df.loc[df['N_S_Part'] > 50, 'N_S_Part'] = 50  # Cap at 50
            outliers_count = (df['N_S_Part'] == 50).sum()
            if outliers_count > 0:
                print(f"  N_S_Part outliers handled: {outliers_count} entries (max: {original_max} -> 50)")
        
        return df
    
    def encode_features(self, df):
        """Encode features"""
        df_encoded = df.copy()
        
        # Label encode categorical features (if needed)
        categorical_features = ['Sex', 'Age', 'Edu_lvl']
        
        for feature in categorical_features:
            if feature in self.selected_features:
                # Skip encoding if already numeric
                if df_encoded[feature].dtype in ['int64', 'float64']:
                    continue
                    
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df_encoded
    
    def split_features_target(self, df):
        """Split features and target"""
        X = df[self.selected_features].values.astype(np.float32)
        y = df[self.target_column].values.astype(np.int32)
        
        return X, y
    
    def preprocess(self, test_size=0.2, random_state=42):
        """Full preprocessing pipeline"""
        print("=" * 50)
        print("Starting HIV/AIDS data preprocessing")
        print("=" * 50)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Clean data
        df_clean = self.clean_data(df)
        
        # 3. Encode features
        df_encoded = self.encode_features(df_clean)
        
        # 4. Split features and target
        X, y = self.split_features_target(df_encoded)
        
        # 5. Print data distribution
        self._print_data_info(X, y)
        
        # 6. Train/test split (with stratification)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # 7. Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Preprocessing complete:")
        print(f"  - Training set: {X_train_scaled.shape}")
        print(f"  - Test set: {X_test_scaled.shape}")
        print(f"  - Positive rate: {np.mean(y):.1%}")
        
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
        """Print dataset info"""
        print(f"\nDataset info:")
        print(f"  - Total samples: {len(X):,}")
        print(f"  - Number of features: {X.shape[1]}")
        print(f"  - Positive samples: {np.sum(y):,} ({np.mean(y):.1%})")
        print(f"  - Negative samples: {len(y) - np.sum(y):,} ({1-np.mean(y):.1%})")
        
        # Basic statistics per feature
        print(f"\nBasic statistics per feature:")
        for i, feature in enumerate(self.selected_features):
            values = X[:, i]
            print(f"  {feature}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

if __name__ == "__main__":
    # Test execution
    data_path = "../../data/HIV_AIDS_DataSet.csv"
    if os.path.exists(data_path):
        preprocessor = HIVDataPreprocessor(data_path)
        result = preprocessor.preprocess()
        print("Preprocessing test completed!")
    else:
        print(f"Test data file not found: {data_path}")
