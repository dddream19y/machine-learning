"""
模型訓練腳本
訓練所有模型並儲存到 models 資料夾
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 設定路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PARENT_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 確保資料夾存在
os.makedirs(os.path.join(MODELS_DIR, 'sklearn'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'tensorflow'), exist_ok=True)


def train_sklearn_models():
    """訓練 Sklearn 模型"""
    print("=" * 50)
    print("訓練 Sklearn 模型...")
    print("=" * 50)
    
    # 載入資料
    df = pd.read_csv(os.path.join(DATA_DIR, 'heart_disease.csv'))
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特徵選取
    selector = SelectKBest(score_func=chi2, k=8)
    X_selected = selector.fit_transform(np.abs(X_scaled), y)
    
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    print(f"選取的特徵: {selected_features}")
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    metrics = {
        'sklearn': {},
        'selected_features': selected_features
    }
    
    # 1. Logistic Regression
    print("\n訓練 Logistic Regression...")
    param_grid_log = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    grid_log = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid_log, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_log.fit(X_train, y_train)
    best_log = grid_log.best_estimator_
    
    y_pred_log = best_log.predict(X_test)
    y_prob_log = best_log.predict_proba(X_test)[:, 1]
    
    log_accuracy = accuracy_score(y_test, y_pred_log)
    log_auc = roc_auc_score(y_test, y_prob_log)
    
    print(f"  準確率: {log_accuracy:.4f}")
    print(f"  ROC AUC: {log_auc:.4f}")
    print(f"  最佳參數: {grid_log.best_params_}")
    
    metrics['sklearn']['logistic_regression'] = {
        'accuracy': float(log_accuracy),
        'roc_auc': float(log_auc),
        'params': grid_log.best_params_
    }
    
    # 2. Random Forest
    print("\n訓練 Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    
    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    
    print(f"  準確率: {rf_accuracy:.4f}")
    print(f"  ROC AUC: {rf_auc:.4f}")
    print(f"  最佳參數: {grid_rf.best_params_}")
    
    metrics['sklearn']['random_forest'] = {
        'accuracy': float(rf_accuracy),
        'roc_auc': float(rf_auc),
        'params': grid_rf.best_params_
    }
    
    # 儲存模型
    sklearn_path = os.path.join(MODELS_DIR, 'sklearn')
    joblib.dump(scaler, os.path.join(sklearn_path, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(sklearn_path, 'selector.pkl'))
    joblib.dump(best_log, os.path.join(sklearn_path, 'logistic_regression.pkl'))
    joblib.dump(best_rf, os.path.join(sklearn_path, 'random_forest.pkl'))
    
    print("\nSklearn 模型已儲存!")
    return metrics


def train_tensorflow_models():
    """訓練 TensorFlow 模型"""
    print("\n" + "=" * 50)
    print("訓練 TensorFlow 模型...")
    print("=" * 50)
    
    # 延遲 import 以避免不必要的載入
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Input, Dense
    from keras.utils import to_categorical
    
    # 載入資料
    df = pd.read_csv(os.path.join(DATA_DIR, 'heart_statlog_cleveland_hungary_final.csv'))
    
    np.random.seed(10)
    dataset = df.values
    np.random.shuffle(dataset)
    
    X = dataset[:, 0:11].astype(np.float32)
    y = dataset[:, 11]
    
    # 儲存正規化參數
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    # 正規化
    X_norm = (X - X_mean) / X_std
    
    # 轉換 y 為 categorical
    y_cat = to_categorical(y)
    
    # 分割資料
    X_train, X_test = X_norm[:690], X_norm[690:]
    y_train, y_test = y_cat[:690], y_cat[690:]
    
    metrics = {'tensorflow': {}}
    training_history = {}
    
    # 1. Model SGD (Binary output)
    print("\n訓練 Model SGD...")
    model_sgd = Sequential([
        Input(shape=(11,)),
        Dense(10, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model_sgd.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history_sgd = model_sgd.fit(X_norm, y[:, np.newaxis], epochs=150, batch_size=10, verbose=0)
    
    loss, accuracy = model_sgd.evaluate(X_norm, y[:, np.newaxis], verbose=0)
    print(f"  準確率: {accuracy:.4f}")
    
    metrics['tensorflow']['model_sgd'] = {'accuracy': float(accuracy)}
    training_history['model_sgd'] = {
        'accuracy': [float(x) for x in history_sgd.history['accuracy']],
        'loss': [float(x) for x in history_sgd.history['loss']]
    }
    
    # 2. Model Adam (Categorical output)
    print("\n訓練 Model Adam...")
    model_adam = Sequential([
        Input(shape=(11,)),
        Dense(10, activation="relu"),
        Dense(8, activation="relu"),
        Dense(2, activation="softmax")
    ])
    model_adam.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history_adam = model_adam.fit(X_norm, y_cat, epochs=150, batch_size=10, verbose=0)
    
    loss, accuracy = model_adam.evaluate(X_norm, y_cat, verbose=0)
    print(f"  準確率: {accuracy:.4f}")
    
    metrics['tensorflow']['model_adam'] = {'accuracy': float(accuracy)}
    training_history['model_adam'] = {
        'accuracy': [float(x) for x in history_adam.history['accuracy']],
        'loss': [float(x) for x in history_adam.history['loss']]
    }
    
    # 3. Model Final (with validation)
    print("\n訓練 Model Final...")
    model_final = Sequential([
        Input(shape=(11,)),
        Dense(10, activation="relu"),
        Dense(8, activation="relu"),
        Dense(2, activation="softmax")
    ])
    model_final.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history_final = model_final.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150, batch_size=10, verbose=0
    )
    
    train_loss, train_accuracy = model_final.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)
    print(f"  訓練準確率: {train_accuracy:.4f}")
    print(f"  測試準確率: {test_accuracy:.4f}")
    
    metrics['tensorflow']['model_final'] = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy)
    }
    training_history['model_final'] = {
        'accuracy': [float(x) for x in history_final.history['accuracy']],
        'loss': [float(x) for x in history_final.history['loss']],
        'val_accuracy': [float(x) for x in history_final.history['val_accuracy']],
        'val_loss': [float(x) for x in history_final.history['val_loss']]
    }
    
    # 儲存模型
    tf_path = os.path.join(MODELS_DIR, 'tensorflow')
    
    # 儲存正規化參數
    np.savez(os.path.join(tf_path, 'normalization_params.npz'), mean=X_mean, std=X_std)
    
    # 儲存模型
    model_sgd.save(os.path.join(tf_path, 'model_sgd.keras'))
    model_adam.save(os.path.join(tf_path, 'model_adam.keras'))
    model_final.save(os.path.join(tf_path, 'model_final.keras'))
    
    # 儲存訓練歷史
    with open(os.path.join(tf_path, 'training_history.json'), 'w') as f:
        json.dump(training_history, f)
    
    print("\nTensorFlow 模型已儲存!")
    return metrics


def main():
    """主程式"""
    print("開始訓練所有模型...")
    print()
    
    # 訓練 Sklearn 模型
    sklearn_metrics = train_sklearn_models()
    
    # 訓練 TensorFlow 模型
    tf_metrics = train_tensorflow_models()
    
    # 合併並儲存指標
    all_metrics = {
        **sklearn_metrics,
        **tf_metrics
    }
    
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print("所有模型訓練完成!")
    print("=" * 50)
    print(f"模型儲存位置: {MODELS_DIR}")


if __name__ == '__main__':
    main()
