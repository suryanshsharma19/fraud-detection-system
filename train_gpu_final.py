import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')  # XGBoost is noisy with warnings

print("Starting fraud detection model training...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FraudNN(nn.Module):
    def __init__(self, input_size):
        super(FraudNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def detect_gpu():
    print("=== GPU Detection ===")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")
        return device
    else:
        print("Error: No GPU - Using CPU")
        return torch.device('cpu')

def load_data():
    print("=== Loading Data ===")
    data_files = [
        ('data/raw/creditcard.csv', 'Class'),
        ('data/raw/PS_20174392719_1491204439457_log.csv', 'isFraud'),
        ('data/raw/creditcard_2023.csv', 'Class')
    ]
    
    all_X, all_y = [], []
    
    for file_path, target_col in data_files:
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            if 'PS_' in file_path:
                df = pd.read_csv(file_path, nrows=300000)  # Sample large dataset
            else:
                df = pd.read_csv(file_path)
            
            if target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Keep numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X = X[numeric_cols].fillna(0)
                
                all_X.append(X.values)
                all_y.append(y.values)
                
                print(f"  {len(df):,} samples, {len(numeric_cols)} features, {y.mean()*100:.1f}% fraud")
    
    # Use minimum features across datasets
    min_features = min(x.shape[1] for x in all_X)
    print(f"Using {min_features} features")
    
    # Standardize feature count
    standardized_X = []
    for X in all_X:
        if X.shape[1] >= min_features:
            standardized_X.append(X[:, :min_features])
        else:
            # Pad with zeros
            padding = np.zeros((X.shape[0], min_features - X.shape[1]))
            standardized_X.append(np.hstack([X, padding]))
    
    # Combine datasets
    X_combined = np.vstack(standardized_X)
    y_combined = np.hstack(all_y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    print(f"Final: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")
    print(f"Fraud rate: {y_combined.mean()*100:.1f}%")
    
    joblib.dump(scaler, 'data/models/gpu_scaler.pkl')
    return X_scaled, y_combined

def train_pytorch_gpu(X_train, y_train, X_test, y_test, device):
    print("=== Training PyTorch Neural Network ===")
    
    # Create datasets (no multiprocessing)
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=0)
    
    # Initialize model
    model = FraudNN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {device}")
    
    # Training loop
    epochs = 10
    best_auc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_targets, val_predictions)
        avg_loss = train_loss / len(train_loader)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': X_train.shape[1],
                'auc_score': val_auc
            }, 'data/models/pytorch_gpu_final.pth')
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss: {avg_loss:.4f}, AUC: {val_auc:.4f}")
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s, Best AUC: {best_auc:.4f}")
    return model, best_auc

def train_xgboost_gpu(X_train, y_train, X_test, y_test):
    print("=== Training XGBoost (GPU) ===")
    
    start_time = time.time()
    try:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        train_time = time.time() - start_time
        print(f"GPU training: {train_time:.1f}s, AUC: {auc_score:.4f}")
        
        joblib.dump(model, 'data/models/xgboost_gpu_final.pkl')
        return model, auc_score
        
    except Exception as e:
        print(f"GPU failed: {e}")
        print("Falling back to CPU...")
        
        model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        train_time = time.time() - start_time
        print(f"CPU fallback: {train_time:.1f}s, AUC: {auc_score:.4f}")
        
        joblib.dump(model, 'data/models/xgboost_cpu_final.pkl')
        return model, auc_score

def main():
    print("GPU-Accelerated Fraud Detection Training")
    print("=" * 50)
    
    # Setup
    device = detect_gpu()
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
    
    os.makedirs('data/models', exist_ok=True)
    results = {}
    
    # Train models
    print("\n" + "=" * 50)
    pytorch_model, pytorch_auc = train_pytorch_gpu(X_train, y_train, X_test, y_test, device)
    results['PyTorch (GPU)'] = pytorch_auc
    
    print("\n" + "=" * 50)
    xgb_model, xgb_auc = train_xgboost_gpu(X_train, y_train, X_test, y_test)
    results['XGBoost (GPU)'] = xgb_auc
    
    # Results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    best_model = max(results, key=results.get)
    for model, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        status = " 🥇" if model == best_model else ""
        print(f"{model}: {auc:.4f} AUC{status}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak GPU memory: {gpu_memory:.2f} GB")
    
    print(f"\nBest model: {best_model} ({results[best_model]:.4f} AUC)")
    print(" Models saved to data/models/")
    print("GPU training complete!")

if __name__ == "__main__":
    main()