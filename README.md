# Heart Disease Prediction - Machine Learning Final Project

使用機器學習方法預測心臟疾病的專案。

## 專案結構

```
machine-learning/
├── data/                          # 資料集
│   ├── heart_disease.csv
│   └── heart_statlog_cleveland_hungary_final.csv
├── sklearn_model/                 # Scikit-learn 模型（原始程式碼）
│   └── main.py                    # Logistic Regression & Random Forest
├── tensorflow_model/              # TensorFlow/Keras 模型（原始程式碼）
│   └── app.py                     # 神經網路模型
├── webapp/                        # 網頁應用程式
│   ├── app.py                     # Flask 主程式
│   ├── train_models.py            # 模型訓練腳本
│   ├── models/                    # 訓練好的模型
│   │   ├── sklearn/               # Sklearn 模型檔案
│   │   └── tensorflow/            # TensorFlow 模型檔案
│   └── templates/                 # HTML 模板
│       ├── base.html              # 基礎模板
│       ├── index.html             # 首頁
│       ├── predict.html           # 預測頁面
│       ├── compare.html           # 模型比較頁面
│       ├── metrics.html           # 效果展示頁面
│       ├── training.html          # 訓練過程頁面
│       └── about.html             # 關於頁面
├── requirements.txt               # 套件依賴
└── README.md
```

## 模型說明

### Sklearn 模型
- **Logistic Regression**: 使用 GridSearchCV 調參
- **Random Forest**: 使用 GridSearchCV 調參
- 特徵選取: SelectKBest (chi2)，選取前 8 個重要特徵

### TensorFlow 模型
- **Model SGD**: SGD 優化器，Binary Crossentropy
- **Model Adam**: Adam 優化器，Categorical Crossentropy
- **Model Final**: Adam 優化器，包含訓練/驗證分割
- 多層感知器架構：11 → 10 → 8 → 2 (softmax)

## 安裝與執行

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 訓練模型
```bash
cd webapp
python train_models.py
```

### 3. 啟動網站
```bash
cd webapp
python app.py
```

然後在瀏覽器開啟 http://localhost:5000

## 網站功能

1. **首頁**: 專案介紹與功能導覽
2. **預測測試**: 輸入健康指標進行心臟病風險預測
3. **模型比較**: 同時比較 5 種模型的預測結果
4. **效果展示**: 查看各模型的準確率、ROC AUC 等指標
5. **訓練過程**: TensorFlow 模型的 Loss/Accuracy 訓練曲線
6. **關於**: 專案說明與技術架構

## 資料集

- `heart_disease.csv`: 11 個特徵的心臟病資料集
- `heart_statlog_cleveland_hungary_final.csv`: Cleveland/Hungary 合併資料集

### 特徵說明
| 特徵 | 說明 |
|------|------|
| age | 年齡 |
| sex | 性別 (1=男, 0=女) |
| chest pain type | 胸痛類型 (1-4) |
| resting bp s | 靜息血壓 (mm Hg) |
| cholesterol | 膽固醇 (mg/dl) |
| fasting blood sugar | 空腹血糖 > 120 mg/dl |
| resting ecg | 靜息心電圖結果 |
| max heart rate | 最大心率 |
| exercise angina | 運動誘發心絞痛 |
| oldpeak | ST 段下降值 |
| ST slope | ST 斜率 |

