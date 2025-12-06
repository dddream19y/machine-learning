# Heart Disease Prediction - Machine Learning Final Project

使用機器學習方法預測心臟疾病的專案。

## 專案結構

```
machine-learning/
├── data/                          # 資料集
│   ├── heart_disease.csv
│   └── heart_statlog_cleveland_hungary_final.csv
├── sklearn_model/                 # Scikit-learn 模型
│   └── main.py                    # Logistic Regression & Random Forest
├── tensorflow_model/              # TensorFlow/Keras 模型
│   └── app.py                     # 神經網路模型
├── requirements.txt               # 套件依賴
└── README.md
```

## 模型說明

### Sklearn 模型 (`sklearn_model/main.py`)
- **Logistic Regression**: 使用 GridSearchCV 調參
- **Random Forest**: 使用 GridSearchCV 調參
- 特徵選取: SelectKBest (chi2)
- 包含模型評估、混淆矩陣、ROC 曲線

### TensorFlow 模型 (`tensorflow_model/app.py`)
- 多層感知器 (MLP) 神經網路
- 包含多種優化器比較 (SGD, Adam)
- 訓練/驗證集分割評估

## 安裝與執行

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行 Sklearn 模型
cd sklearn_model
python main.py

# 執行 TensorFlow 模型
cd tensorflow_model
python app.py
```

## 資料集

- `heart_disease.csv`: 11 個特徵的心臟病資料集
- `heart_statlog_cleveland_hungary_final.csv`: Cleveland/Hungary 合併資料集
