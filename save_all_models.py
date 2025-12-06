"""
執行此腳本來儲存所有模型供網頁應用程式使用
此腳本會：
1. 執行 sklearn_model/main.py 中的 save_models_for_webapp()
2. 執行 tensorflow_model/app.py 中的 save_models_for_webapp()
"""

import os
import sys

# 確保在正確的目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("="*60)
print("開始儲存所有模型供網頁應用程式使用")
print("="*60)

# ============================================================
# Part 1: 儲存 Sklearn 模型
# ============================================================
print("\n" + "="*60)
print("Part 1: 儲存 Sklearn 模型")
print("="*60)

os.chdir(os.path.join(BASE_DIR, 'sklearn_model'))
print(f"工作目錄: {os.getcwd()}")

# 執行 sklearn 模型檔案（這會訓練模型並建立所有變數）
exec(open('main.py', encoding='utf-8').read())

# 然後執行儲存函式
print("\n--- 執行 save_models_for_webapp() ---")
save_models_for_webapp()

print("\nSklearn 模型儲存完成！")

# ============================================================
# Part 2: 儲存 TensorFlow 模型
# ============================================================
print("\n" + "="*60)
print("Part 2: 儲存 TensorFlow 模型")
print("="*60)

os.chdir(os.path.join(BASE_DIR, 'tensorflow_model'))
print(f"工作目錄: {os.getcwd()}")

# 由於 TensorFlow 模型檔案有繪圖，我們需要先設定 matplotlib 為非互動模式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 清除之前的變數（避免衝突）
plt.close('all')

# 執行 tensorflow 模型檔案
exec(open('app.py', encoding='utf-8').read())

# 然後執行儲存函式
print("\n--- 執行 save_models_for_webapp() ---")
save_models_for_webapp()

print("\nTensorFlow 模型儲存完成！")

# ============================================================
# 完成
# ============================================================
print("\n" + "="*60)
print("所有模型已儲存完成！")
print("="*60)
print("\n儲存的檔案：")
print("\nsklearn_model/:")
for f in ['scaler.pkl', 'selector.pkl', 'logistic_regression.pkl', 
          'random_forest.pkl', 'sklearn_metrics.json']:
    path = os.path.join(BASE_DIR, 'sklearn_model', f)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"  {status} {f}")

print("\ntensorflow_model/:")
for f in ['normalization_params.npz', 'model_sgd.keras', 'model_adam.keras',
          'model_final.keras', 'training_history.json', 'tf_metrics.json']:
    path = os.path.join(BASE_DIR, 'tensorflow_model', f)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"  {status} {f}")

print("\n現在可以啟動網頁應用程式了！")
print("執行: cd webapp && python app.py")
