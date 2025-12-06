"""
在 tensorflow_model 目錄中執行此腳本來儲存模型
"""
import os
import sys

# 設定 matplotlib 為非互動模式（避免彈出視窗）
import matplotlib
matplotlib.use('Agg')

# 確保工作目錄正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"工作目錄: {os.getcwd()}")

# 執行 app.py 的所有內容
print("="*50)
print("執行 app.py 來訓練模型...")
print("="*50)

# 使用 exec 來執行整個檔案，這樣可以存取所有變數
with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 執行程式碼
exec(code)

# 現在呼叫 save_models_for_webapp
print("\n" + "="*50)
print("儲存模型供網頁應用程式使用...")
print("="*50)
save_models_for_webapp()

print("\n完成！")
