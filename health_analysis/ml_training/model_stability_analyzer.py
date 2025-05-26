# model_stability_analyzer.py
# Analyze model stability through 100 repeated training sessions
# Measure mean accuracy and variance of machine learning models

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
import warnings
from pathlib import Path
import time
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Model hyperparameter configurations
MODEL_CONFIG = {
    'logistic_regression': {
        'param_grid': {
            'C': [0.1, 1, 10],
            'max_iter': [1000],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced']
        },
        'cv_folds': 3,
        'scoring': 'roc_auc',
        'random_state': None  # Will be set dynamically
    }
}

class HIVDataPreprocessor:
    def __init__(self, data_path):
        """Preprocessor for the HIV/AIDS dataset"""
        self.data_path = data_path
        
        # 18 selected features (matching your updated code)
        self.selected_features = [
            'Sex', 'Age', 'Edu_lvl', 'Had_Sex', 'N_S_Part', 
            'Con_Use', 'R_Use_Con', 'R_SeA', 'R_Have_1SP', 
            'R_Nhave_Sex', 'HIV_Mosq', 'H_STI', 'H_O_STI', 
            'E_T_HIV', 'P_T_HIV', 'S_Test', 'T_in_LAB', 'H_AIDS'
        ]
        
        # Target variable (matching your updated code)
        self.target_column = 'F_T_Resu'
    
    def preprocess(self, test_size=0.2, random_state=42):
        """Full preprocessing pipeline"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Select only necessary columns
            df_clean = df[self.selected_features + [self.target_column]].copy()
            
            # Handle missing values
            for col in self.selected_features:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            df_clean[self.target_column] = pd.to_numeric(df_clean[self.target_column], errors='coerce').fillna(0)
            
            # Handle outliers in N_S_Part
            if 'N_S_Part' in df_clean.columns:
                df_clean.loc[df_clean['N_S_Part'] > 50, 'N_S_Part'] = 50
            
            X = df_clean[self.selected_features].values.astype(np.float32)
            y = df_clean[self.target_column].values.astype(np.int32)
            
            # Train/test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y, shuffle=True
            )
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'features': self.selected_features,
                'X_full': X, 'y_full': y
            }
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise

def train_single_model_iteration(args):
    """Train a single model iteration - designed for multiprocessing"""
    model_name, model_config, X_full, y_full, test_size, iteration_seed = args
    
    try:
        # Split data with unique random seed
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, 
            test_size=test_size, 
            random_state=iteration_seed,
            stratify=y_full
        )
        
        # Create model with specific random seed
        if model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=iteration_seed)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            model_config['param_grid'],
            cv=model_config['cv_folds'],
            scoring=model_config['scoring'],
            n_jobs=1  # Important for multiprocessing
        )
        
        # Train model
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model and make predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': training_time,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        return iteration_seed, metrics
        
    except Exception as e:
        print(f"Error in iteration {iteration_seed}: {e}")
        return iteration_seed, None

class ModelStabilityAnalyzer:
    """Analyze model stability through repeated training sessions"""
    
    def __init__(self, data_path, output_dir="stability_analysis_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize preprocessor
        self.preprocessor = HIVDataPreprocessor(data_path)
        
        # Results storage
        self.results = {}
        
    def run_stability_analysis(self, num_iterations=100, test_size=0.2, use_multiprocessing=True):
        """Run stability analysis with repeated training"""
        print(f"🔬 Starting Model Stability Analysis ({num_iterations} iterations)")
        print("=" * 80)
        
        # Load and preprocess data once
        print("📊 Loading and preprocessing data...")
        data = self.preprocessor.preprocess(test_size=test_size, random_state=42)
        X_full = data['X_full']
        y_full = data['y_full']
        
        print(f"Dataset Information:")
        print(f"  - Total samples: {len(X_full):,}")
        print(f"  - Features: {X_full.shape[1]}")
        print(f"  - Positive rate: {np.mean(y_full):.1%}")
        print(f"  - Test size: {test_size:.0%}")
        print(f"  - Training iterations: {num_iterations}")
        print()
        
        # Models to test
        models_to_test = ['LogisticRegression']
        
        # Run analysis for each model
        for model_name in models_to_test:
            print(f"🤖 Analyzing {model_name} stability...")
            
            model_config = MODEL_CONFIG[model_name.lower().replace('regression', '_regression')]
            
            # Prepare arguments for multiprocessing
            args_list = []
            for i in range(num_iterations):
                iteration_seed = np.random.randint(0, 100000)
                args_list.append((
                    model_name, model_config, X_full, y_full, test_size, iteration_seed
                ))
            
            # Execute training iterations
            if use_multiprocessing and num_iterations > 10:
                results = self._run_multiprocessing(args_list, model_name)
            else:
                results = self._run_sequential(args_list, model_name)
            
            # Process and store results
            self._process_model_results(model_name, results, num_iterations)
        
        # Analyze and save final results
        self.analyze_and_save_results(num_iterations)
        
        return self.results
    
    def _run_multiprocessing(self, args_list, model_name):
        """Run training with multiprocessing"""
        num_cores = min(mp.cpu_count(), 8)  # Limit cores to avoid overwhelming
        results = []
        
        print(f"  Using {num_cores} CPU cores for parallel training...")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all jobs
            future_to_args = {
                executor.submit(train_single_model_iteration, args): args 
                for args in args_list
            }
            
            # Collect results with progress bar
            with tqdm(total=len(args_list), desc=f"  {model_name}") as pbar:
                for future in as_completed(future_to_args):
                    try:
                        iteration_seed, metrics = future.result()
                        if metrics is not None:
                            results.append((iteration_seed, metrics))
                        pbar.update(1)
                    except Exception as e:
                        print(f"    Warning: Failed iteration - {e}")
                        pbar.update(1)
        
        return results
    
    def _run_sequential(self, args_list, model_name):
        """Run training sequentially"""
        results = []
        
        for args in tqdm(args_list, desc=f"  {model_name}"):
            iteration_seed, metrics = train_single_model_iteration(args)
            if metrics is not None:
                results.append((iteration_seed, metrics))
        
        return results
    
    def _process_model_results(self, model_name, results, num_iterations):
        """Process and organize results for a single model"""
        if not results:
            print(f"  ❌ No successful results for {model_name}")
            return
        
        # Organize metrics
        model_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'training_time': [],
            'cv_score': [],
            'iteration_seeds': [],
            'best_params': []
        }
        
        for iteration_seed, metrics in results:
            model_results['iteration_seeds'].append(iteration_seed)
            model_results['best_params'].append(metrics['best_params'])
            
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time', 'cv_score']:
                model_results[metric_name].append(metrics[metric_name])
        
        # Store results
        self.results[model_name] = model_results
        
        # Print immediate summary
        successful_runs = len(results)
        accuracy_mean = np.mean(model_results['accuracy'])
        accuracy_std = np.std(model_results['accuracy'])
        
        print(f"  ✅ {model_name} completed: {successful_runs}/{num_iterations} successful runs")
        print(f"     Mean Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
        print()
    
    def analyze_and_save_results(self, num_iterations):
        """Analyze and save comprehensive results"""
        print("=" * 80)
        print("📈 STABILITY ANALYSIS RESULTS")
        print("=" * 80)
        
        analysis_results = {}
        comparison_data = []
        
        # Analyze each model
        for model_name, results in self.results.items():
            print(f"\n🤖 {model_name} Analysis ({len(results['accuracy'])} successful runs):")
            print("-" * 60)
            
            model_stats = {}
            
            # Calculate statistics for each metric
            for metric_name, values in results.items():
                if metric_name in ['iteration_seeds', 'best_params']:
                    continue
                
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    median_val = np.median(values)
                    cv = std_val / mean_val if mean_val != 0 else 0.0  # Coefficient of variation
                    
                    model_stats[metric_name] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'min': float(min_val),
                        'max': float(max_val),
                        'median': float(median_val),
                        'cv': float(cv),
                        'values': [float(v) for v in values]
                    }
                    
                    if metric_name == 'training_time':
                        print(f"  ⏱️  {metric_name}: {mean_val:.3f}s (±{std_val:.3f}s)")
                    else:
                        print(f"  📊 {metric_name}: {mean_val:.4f} (±{std_val:.4f}) "
                              f"Range: [{min_val:.4f}, {max_val:.4f}] CV: {cv:.1%}")
            
            analysis_results[model_name] = model_stats
            
            # Store comparison data
            if 'accuracy' in model_stats:
                comparison_data.append({
                    'model': model_name,
                    'accuracy_mean': model_stats['accuracy']['mean'],
                    'accuracy_std': model_stats['accuracy']['std'],
                    'roc_auc_mean': model_stats.get('roc_auc', {}).get('mean', 0),
                    'stability_score': model_stats['accuracy']['mean'] / (1 + model_stats['accuracy']['std'])
                })
        
        # Model comparison
        print(f"\n🏆 Model Comparison (based on accuracy):")
        print("-" * 60)
        
        comparison_data.sort(key=lambda x: x['accuracy_mean'], reverse=True)
        
        for i, model_info in enumerate(comparison_data, 1):
            print(f"  {i}. {model_info['model']}: "
                  f"Accuracy {model_info['accuracy_mean']:.4f} (±{model_info['accuracy_std']:.4f}) | "
                  f"AUC {model_info['roc_auc_mean']:.4f} | "
                  f"Stability Score {model_info['stability_score']:.4f}")
        
        # Best model selection
        if comparison_data:
            best_accuracy_model = comparison_data[0]
            best_stable_model = max(comparison_data, key=lambda x: x['stability_score'])
            
            print(f"\n🥇 Highest Accuracy: {best_accuracy_model['model']} "
                  f"({best_accuracy_model['accuracy_mean']:.4f})")
            print(f"🎯 Most Stable: {best_stable_model['model']} "
                  f"(Stability Score: {best_stable_model['stability_score']:.4f})")
        
        # Save results
        self._save_comprehensive_results(analysis_results, comparison_data, num_iterations)
        
        # Create visualizations
        self._create_visualizations(num_iterations)
        
        return analysis_results
    
    def _save_comprehensive_results(self, analysis_results, comparison_data, num_iterations):
        """Save comprehensive results to multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save JSON results
        json_filename = self.output_dir / f"stability_analysis_{num_iterations}runs_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_info': {
                    'num_iterations': num_iterations,
                    'timestamp': timestamp,
                    'data_path': str(self.data_path)
                },
                'detailed_results': analysis_results,
                'model_comparison': comparison_data,
                'raw_results': {k: {metric: values for metric, values in v.items() 
                               if metric not in ['iteration_seeds', 'best_params']} 
                               for k, v in self.results.items()}
            }, f, indent=2, ensure_ascii=False)
        
        # 2. Save CSV summary
        summary_data = []
        for model_name, stats in analysis_results.items():
            row = {'Model': model_name}
            for metric_name, metric_stats in stats.items():
                if isinstance(metric_stats, dict):
                    row[f'{metric_name}_mean'] = metric_stats['mean']
                    row[f'{metric_name}_std'] = metric_stats['std']
                    row[f'{metric_name}_cv'] = metric_stats['cv']
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_filename = self.output_dir / f"model_stability_summary_{timestamp}.csv"
        summary_df.to_csv(csv_filename, index=False)
        
        # 3. Save detailed CSV for each model
        for model_name, results in self.results.items():
            detailed_data = []
            for i in range(len(results['accuracy'])):
                row = {'iteration': i + 1}
                for metric_name, values in results.items():
                    if metric_name not in ['iteration_seeds', 'best_params']:
                        row[metric_name] = values[i]
                detailed_data.append(row)
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv = self.output_dir / f"{model_name}_detailed_results_{timestamp}.csv"
            detailed_df.to_csv(detailed_csv, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"  📊 Summary JSON: {json_filename.name}")
        print(f"  📈 Summary CSV: {csv_filename.name}")
        print(f"  📋 Detailed CSVs: {len(self.results)} files")
    
    def _create_visualizations(self, num_iterations):
        """Create comprehensive visualizations"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Stability Analysis ({num_iterations} iterations)', fontsize=16, fontweight='bold')
        
        # Accuracy boxplot
        accuracy_data = []
        for model_name, results in self.results.items():
            for acc in results['accuracy']:
                accuracy_data.append({'Model': model_name, 'Accuracy': acc})
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            sns.boxplot(data=accuracy_df, x='Model', y='Accuracy', ax=axes[0,0])
            axes[0,0].set_title('Accuracy Distribution by Model')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # ROC AUC comparison
        auc_data = []
        for model_name, results in self.results.items():
            for auc in results['roc_auc']:
                auc_data.append({'Model': model_name, 'ROC_AUC': auc})
        
        if auc_data:
            auc_df = pd.DataFrame(auc_data)
            sns.boxplot(data=auc_df, x='Model', y='ROC_AUC', ax=axes[0,1])
            axes[0,1].set_title('ROC AUC Distribution by Model')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        time_data = []
        for model_name, results in self.results.items():
            for time_val in results['training_time']:
                time_data.append({'Model': model_name, 'Training_Time': time_val})
        
        if time_data:
            time_df = pd.DataFrame(time_data)
            sns.boxplot(data=time_df, x='Model', y='Training_Time', ax=axes[1,0])
            axes[1,0].set_title('Training Time Distribution by Model')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs Training Time scatter
        scatter_data = []
        for model_name, results in self.results.items():
            for i in range(len(results['accuracy'])):
                scatter_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'][i],
                    'Training_Time': results['training_time'][i]
                })
        
        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            for model_name in scatter_df['Model'].unique():
                model_data = scatter_df[scatter_df['Model'] == model_name]
                axes[1,1].scatter(model_data['Training_Time'], model_data['Accuracy'], 
                                label=model_name, alpha=0.6)
            axes[1,1].set_xlabel('Training Time (seconds)')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_title('Accuracy vs Training Time')
            axes[1,1].legend()
        
        plt.tight_layout()
        viz_filename = self.output_dir / f"stability_analysis_plots_{timestamp}.png"
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Individual model performance over iterations
        if len(self.results) > 0:
            fig, axes = plt.subplots(len(self.results), 1, figsize=(12, 4*len(self.results)))
            if len(self.results) == 1:
                axes = [axes]
            
            for idx, (model_name, results) in enumerate(self.results.items()):
                iterations = range(1, len(results['accuracy']) + 1)
                
                axes[idx].plot(iterations, results['accuracy'], 'b-', alpha=0.7, label='Accuracy')
                axes[idx].plot(iterations, results['roc_auc'], 'r-', alpha=0.7, label='ROC AUC')
                
                # Add mean lines
                acc_mean = np.mean(results['accuracy'])
                auc_mean = np.mean(results['roc_auc'])
                axes[idx].axhline(y=acc_mean, color='blue', linestyle='--', alpha=0.5)
                axes[idx].axhline(y=auc_mean, color='red', linestyle='--', alpha=0.5)
                
                axes[idx].set_title(f'{model_name} Performance Over Iterations')
                axes[idx].set_xlabel('Iteration')
                axes[idx].set_ylabel('Score')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            iteration_plot = self.output_dir / f"iteration_performance_{timestamp}.png"
            plt.savefig(iteration_plot, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  📊 Visualization: {viz_filename.name}")
        print(f"  📈 Iteration plot: {iteration_plot.name}")

# Main execution function
def main():
    """Main function to run stability analysis"""
    print("🏥 HIV/AIDS Model Stability Analyzer")
    print("=" * 80)
    
    # Configuration
    data_path = "../../data/HIV_AIDS_DataSet.csv"  
    num_iterations = 100
    test_size = 0.2
    use_multiprocessing = True
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable with the correct path to your HIV_AIDS_DataSet.csv file")
        return
    
    try:
        # Initialize analyzer
        analyzer = ModelStabilityAnalyzer(
            data_path=data_path,
            output_dir="stability_analysis_results"
        )
        
        # Run stability analysis
        results = analyzer.run_stability_analysis(
            num_iterations=num_iterations,
            test_size=test_size,
            use_multiprocessing=use_multiprocessing
        )
        
        print(f"\n✨ Stability analysis completed successfully!")
        print(f"📁 Results saved in: {analyzer.output_dir}")
        
        # Print final summary
        print(f"\n📋 Final Summary:")
        for model_name, model_results in results.items():
            if 'accuracy' in model_results and len(model_results['accuracy']) > 0:
                acc_mean = np.mean(model_results['accuracy'])
                acc_std = np.std(model_results['accuracy'])
                print(f"  {model_name}: {acc_mean:.4f} ± {acc_std:.4f} accuracy")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()