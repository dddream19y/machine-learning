"""
Heart Disease Prediction Web Application
整合 Sklearn 和 TensorFlow 模型的心臟病預測網站
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
from tensorflow import keras

app = Flask(__name__)

# 設定路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PARENT_DIR, 'data')

# 原始模型路徑（使用 sklearn_model 和 tensorflow_model 資料夾中的模型）
SKLEARN_MODEL_DIR = os.path.join(PARENT_DIR, 'sklearn_model')
TENSORFLOW_MODEL_DIR = os.path.join(PARENT_DIR, 'tensorflow_model')

# 特徵名稱
FEATURE_NAMES = [
    'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    'fasting blood sugar', 'resting ecg', 'max heart rate',
    'exercise angina', 'oldpeak', 'ST slope'
]

# 特徵說明（用於前端顯示）
FEATURE_INFO = {
    'age': {'label': '年齡', 'type': 'number', 'min': 1, 'max': 120, 'default': 50},
    'sex': {'label': '性別', 'type': 'select', 'options': [{'value': 1, 'label': '男性'}, {'value': 0, 'label': '女性'}], 'default': 1},
    'chest pain type': {'label': '胸痛類型', 'type': 'select', 'options': [
        {'value': 1, 'label': '典型心絞痛'},
        {'value': 2, 'label': '非典型心絞痛'},
        {'value': 3, 'label': '非心絞痛'},
        {'value': 4, 'label': '無症狀'}
    ], 'default': 1},
    'resting bp s': {'label': '靜息血壓 (mm Hg)', 'type': 'number', 'min': 80, 'max': 200, 'default': 120},
    'cholesterol': {'label': '膽固醇 (mg/dl)', 'type': 'number', 'min': 100, 'max': 600, 'default': 200},
    'fasting blood sugar': {'label': '空腹血糖 > 120 mg/dl', 'type': 'select', 'options': [{'value': 1, 'label': '是'}, {'value': 0, 'label': '否'}], 'default': 0},
    'resting ecg': {'label': '靜息心電圖', 'type': 'select', 'options': [
        {'value': 0, 'label': '正常'},
        {'value': 1, 'label': 'ST-T 異常'},
        {'value': 2, 'label': '左心室肥大'}
    ], 'default': 0},
    'max heart rate': {'label': '最大心率', 'type': 'number', 'min': 60, 'max': 220, 'default': 150},
    'exercise angina': {'label': '運動誘發心絞痛', 'type': 'select', 'options': [{'value': 1, 'label': '是'}, {'value': 0, 'label': '否'}], 'default': 0},
    'oldpeak': {'label': 'ST 段下降值', 'type': 'number', 'min': -5, 'max': 10, 'step': 0.1, 'default': 1.0},
    'ST slope': {'label': 'ST 斜率', 'type': 'select', 'options': [
        {'value': 1, 'label': '上升'},
        {'value': 2, 'label': '平坦'},
        {'value': 3, 'label': '下降'}
    ], 'default': 2}
}


class ModelManager:
    """管理所有模型的類別"""
    
    def __init__(self):
        self.sklearn_models = {}
        self.tf_models = {}
        self.scaler = None
        self.selector = None
        self.tf_scaler_mean = None
        self.tf_scaler_std = None
        self.model_metrics = {}
        
    def load_sklearn_models(self):
        """載入 Sklearn 模型（從 sklearn_model 資料夾）"""
        try:
            # 載入 scaler 和 selector
            self.scaler = joblib.load(os.path.join(SKLEARN_MODEL_DIR, 'scaler.pkl'))
            self.selector = joblib.load(os.path.join(SKLEARN_MODEL_DIR, 'selector.pkl'))
            
            # 載入 Logistic Regression 模型
            self.sklearn_models['logistic_regression'] = joblib.load(
                os.path.join(SKLEARN_MODEL_DIR, 'logistic_regression.pkl')
            )
            
            # 載入 Random Forest 模型
            self.sklearn_models['random_forest'] = joblib.load(
                os.path.join(SKLEARN_MODEL_DIR, 'random_forest.pkl')
            )
            
            print("Sklearn 模型載入成功（從 sklearn_model 資料夾）")
            return True
        except Exception as e:
            print(f"載入 Sklearn 模型失敗: {e}")
            return False
    
    def load_tf_models(self):
        """載入 TensorFlow 模型（從 tensorflow_model 資料夾）"""
        try:
            # 載入正規化參數
            norm_params = np.load(os.path.join(TENSORFLOW_MODEL_DIR, 'normalization_params.npz'))
            self.tf_scaler_mean = norm_params['mean']
            self.tf_scaler_std = norm_params['std']
            
            # 載入三個 TensorFlow 模型
            self.tf_models['tf_model_sgd'] = keras.models.load_model(
                os.path.join(TENSORFLOW_MODEL_DIR, 'model_sgd.keras')
            )
            self.tf_models['tf_model_adam'] = keras.models.load_model(
                os.path.join(TENSORFLOW_MODEL_DIR, 'model_adam.keras')
            )
            self.tf_models['tf_model_final'] = keras.models.load_model(
                os.path.join(TENSORFLOW_MODEL_DIR, 'model_final.keras')
            )
            
            print("TensorFlow 模型載入成功（從 tensorflow_model 資料夾）")
            return True
        except Exception as e:
            print(f"載入 TensorFlow 模型失敗: {e}")
            return False
    
    def load_metrics(self):
        """載入模型評估指標（從原始模型資料夾）"""
        try:
            # 載入 sklearn 指標
            sklearn_metrics_path = os.path.join(SKLEARN_MODEL_DIR, 'sklearn_metrics.json')
            if os.path.exists(sklearn_metrics_path):
                with open(sklearn_metrics_path, 'r', encoding='utf-8') as f:
                    sklearn_metrics = json.load(f)
                self.model_metrics.update(sklearn_metrics)
            
            # 載入 tensorflow 指標
            tf_metrics_path = os.path.join(TENSORFLOW_MODEL_DIR, 'tf_metrics.json')
            if os.path.exists(tf_metrics_path):
                with open(tf_metrics_path, 'r', encoding='utf-8') as f:
                    tf_metrics = json.load(f)
                self.model_metrics.update(tf_metrics)
            
            print("模型指標載入成功")
            return True
        except Exception as e:
            print(f"載入模型指標失敗: {e}")
            return False
    
    def predict_sklearn(self, model_name, features):
        """使用 Sklearn 模型進行預測"""
        if model_name not in self.sklearn_models:
            return None, None
        
        input_array = np.array(features).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        input_selected = self.selector.transform(np.abs(input_scaled))
        
        model = self.sklearn_models[model_name]
        prediction = model.predict(input_selected)[0]
        probability = model.predict_proba(input_selected)[0][1]
        
        return int(prediction), float(probability)
    
    def predict_tensorflow(self, model_name, features):
        """使用 TensorFlow 模型進行預測"""
        if model_name not in self.tf_models:
            return None, None
        
        input_array = np.array(features).reshape(1, -1).astype(np.float32)
        
        # 正規化
        input_normalized = (input_array - self.tf_scaler_mean) / self.tf_scaler_std
        
        model = self.tf_models[model_name]
        prediction_proba = model.predict(input_normalized, verbose=0)
        
        # 處理不同輸出格式
        if prediction_proba.shape[1] == 2:
            probability = prediction_proba[0][1]
            prediction = 1 if probability > 0.5 else 0
        else:
            probability = prediction_proba[0][0]
            prediction = 1 if probability > 0.5 else 0
        
        return int(prediction), float(probability)


# 初始化模型管理器
model_manager = ModelManager()


@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """預測頁面"""
    if request.method == 'GET':
        return render_template('predict.html', features=FEATURE_INFO)
    
    # POST 請求處理預測
    try:
        data = request.get_json()
        features = [float(data.get(name, 0)) for name in FEATURE_NAMES]
        model_type = data.get('model_type', 'logistic_regression')
        
        # 根據模型類型進行預測
        if model_type in ['logistic_regression', 'random_forest']:
            prediction, probability = model_manager.predict_sklearn(model_type, features)
        else:
            prediction, probability = model_manager.predict_tensorflow(model_type, features)
        
        if prediction is None:
            return jsonify({'error': '模型未載入'}), 500
        
        result = {
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'result_text': '有心臟病風險' if prediction == 1 else '無心臟病風險',
            'model_used': model_type
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare')
def compare():
    """模型比較頁面"""
    return render_template('compare.html', metrics=model_manager.model_metrics)


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """API: 所有模型同時預測並比較"""
    try:
        data = request.get_json()
        features = [float(data.get(name, 0)) for name in FEATURE_NAMES]
        
        results = {}
        
        # Sklearn 模型預測
        for model_name in model_manager.sklearn_models.keys():
            pred, prob = model_manager.predict_sklearn(model_name, features)
            if pred is not None:
                results[model_name] = {
                    'prediction': pred,
                    'probability': round(prob * 100, 2),
                    'result_text': '有風險' if pred == 1 else '無風險'
                }
        
        # TensorFlow 模型預測
        for model_name in model_manager.tf_models.keys():
            pred, prob = model_manager.predict_tensorflow(model_name, features)
            if pred is not None:
                results[model_name] = {
                    'prediction': pred,
                    'probability': round(prob * 100, 2),
                    'result_text': '有風險' if pred == 1 else '無風險'
                }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """模型效果展示頁面"""
    return render_template('metrics.html', metrics=model_manager.model_metrics)


@app.route('/api/metrics')
def api_metrics():
    """API: 獲取模型指標"""
    return jsonify(model_manager.model_metrics)


@app.route('/training')
def training():
    """TensorFlow 訓練過程頁面"""
    return render_template('training.html')


@app.route('/api/training-history')
def api_training_history():
    """API: 獲取 TensorFlow 訓練歷史"""
    try:
        history_path = os.path.join(TENSORFLOW_MODEL_DIR, 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """關於頁面"""
    return render_template('about.html')


def init_app():
    """初始化應用程式"""
    # 載入所有模型
    model_manager.load_sklearn_models()
    model_manager.load_tf_models()
    model_manager.load_metrics()


if __name__ == '__main__':
    init_app()
    app.run(debug=True, port=5000)
