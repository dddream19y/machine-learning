import pandas as pd
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
import matplotlib.pyplot as plt


df = pd.read_csv("../data/heart_statlog_cleveland_hungary_final.csv")

df.head()
df.shape
print(df.info())
df.isnull().sum()

def draw_loss(history):
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "r", label="Validation Loss")
        plt.title("Training and Validation Loss")
    else:
        plt.title("Training Loss")
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def draw_acc(history):
    acc = history.history["accuracy"]
    epochs = range(1, len(acc) + 1)
    if "val_accuracy" in history.history:
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, val_acc, "r--", label="Validation Acc")
        plt.title("Training and Validation Accuracy")
    else:
        plt.title("Training Accuracy")

    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:11]
y = dataset[:, 11]
# 定義模型
model = Sequential()
model.add(Input(shape=(11,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊

# 編譯模型
model.compile(loss="binary_crossentropy", 
              optimizer="sgd", metrics=["accuracy"])
# 訓練模型
history = model.fit(X, y, epochs=150, batch_size=10)

# 評估模型
loss, accuracy = model.evaluate(X, y)
print("準確度 = {:.2f}".format(accuracy))

draw_loss(history)  
draw_acc(history) 

X -= X.mean(axis=0)
X /= X.std(axis=0)

model2 = Sequential()
model2.add(Input(shape=(11,)))
model2.add(Dense(10, activation="relu"))
model2.add(Dense(8, activation="relu"))
model2.add(Dense(1, activation="sigmoid"))
# model.summary()   # 顯示模型摘要資訊
# 編譯模型
model2.compile(loss="binary_crossentropy", 
               optimizer="sgd", metrics=["accuracy"])
# 訓練模型
history2 = model2.fit(X, y, epochs=150, batch_size=10)
# 評估模型
loss, accuracy = model2.evaluate(X, y)
print("準確度 = {:.2f}".format(accuracy))

draw_loss(history2)
draw_acc(history2)

from keras.utils import to_categorical

y
y = to_categorical(y)
y
# 定義模型
model3 = Sequential()
model3.add(Input(shape=(11,)))
model3.add(Dense(10, activation="relu"))
model3.add(Dense(8, activation="relu"))
model3.add(Dense(2, activation="softmax"))
model3.summary() 

model3.compile(loss="categorical_crossentropy", 
               optimizer="sgd", metrics=["accuracy"])

history3 = model3.fit(X, y, epochs=150, batch_size=10)

loss, accuracy = model3.evaluate(X, y)
print("準確度 = {:.2f}".format(accuracy))

draw_acc(history3)
draw_loss(history3)


model4 = Sequential()
model4.add(Input(shape=(11,)))
model4.add(Dense(10, activation="relu"))
model4.add(Dense(8, activation="relu"))
model4.add(Dense(2, activation="softmax"))
model4.summary()   # 顯示模型摘要資訊
# 編譯模型
model4.compile(loss="categorical_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history4 = model4.fit(X, y, epochs=150, batch_size=10)
# 評估模型
loss, accuracy = model4.evaluate(X, y, verbose=0)
print("準確度 = {:.2f}".format(accuracy))
draw_acc(history4)
draw_loss(history4)

X_train, y_train = X[:690], y[:690]  
X_test, y_test = X[690:], y[690:]  
model5 = Sequential()
model5.add(Input(shape=(11,)))
model5.add(Dense(10, activation="relu"))
model5.add(Dense(8, activation="relu"))
model5.add(Dense(2, activation="softmax"))
model5.summary()   # 顯示模型摘要資訊
# 編譯模型
model5.compile(loss="categorical_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history5 = model5.fit(X_train, y_train, epochs=150, batch_size=10)
# 評估模型
loss, accuracy = model5.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model5.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))


model5 = Sequential()
model5.add(Input(shape=(11,)))
model5.add(Dense(10, activation="relu"))
model5.add(Dense(8, activation="relu"))
model5.add(Dense(2, activation="softmax"))
# model5.summary()   # 顯示模型摘要資訊
# 編譯模型
model5.compile(loss="categorical_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history6 = model5.fit(
    X_train, y_train, validation_data=(X_test, y_test), 
    epochs=150, batch_size=10
)
# 評估模型
loss, accuracy = model5.evaluate(X_train, y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model5.evaluate(X_test, y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
draw_acc(history6)
draw_loss(history6)


model5 = Sequential()
model5.add(Input(shape=(11,)))
model5.add(Dense(10, activation="relu"))
model5.add(Dense(8, activation="relu"))
model5.add(Dense(2, activation="softmax"))
model5.summary()   # 顯示模型摘要資訊
# 編譯模型
model5.compile(loss="categorical_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history7 = model5.fit(
    X_train, y_train, validation_data=(X_test, y_test), 
    epochs=10, batch_size=10
)
# 評估模型
loss, accuracy = model5.evaluate(X_train, y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model5.evaluate(X_test, y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
draw_acc(history7)
draw_loss(history7)


# ============================================================
# 以下為網頁應用程式所需的模型儲存區塊
# 此區塊不影響上方原本的程式碼邏輯
# ============================================================

def save_models_for_webapp():
    """
    儲存所有 TensorFlow 模型供網頁應用程式使用
    執行此函數將模型儲存到當前目錄
    """
    import json
    
    # 重新載入資料並訓練模型（確保使用一致的資料）
    np.random.seed(10)
    df_data = pd.read_csv("../data/heart_statlog_cleveland_hungary_final.csv")
    dataset_values = df_data.values
    np.random.shuffle(dataset_values)
    
    X_data = dataset_values[:, 0:11].astype(np.float32)
    y_data = dataset_values[:, 11]
    
    # 儲存正規化參數
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    np.savez('normalization_params.npz', mean=X_mean, std=X_std)
    print("已儲存: normalization_params.npz")
    
    # 正規化資料
    X_norm = (X_data - X_mean) / X_std
    
    from keras.utils import to_categorical
    y_cat = to_categorical(y_data)
    
    # 分割資料
    X_tr, y_tr = X_norm[:690], y_cat[:690]
    X_te, y_te = X_norm[690:], y_cat[690:]
    
    # 訓練並儲存 Model SGD (Binary output, 未正規化資料)
    print("\n訓練 Model SGD...")
    model_sgd = Sequential()
    model_sgd.add(Input(shape=(11,)))
    model_sgd.add(Dense(10, activation="relu"))
    model_sgd.add(Dense(8, activation="relu"))
    model_sgd.add(Dense(1, activation="sigmoid"))
    model_sgd.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    hist_sgd = model_sgd.fit(X_data, y_data, epochs=150, batch_size=10, verbose=0)
    loss_sgd, acc_sgd = model_sgd.evaluate(X_data, y_data, verbose=0)
    model_sgd.save('model_sgd.keras')
    print(f"已儲存: model_sgd.keras (準確度: {acc_sgd:.4f})")
    
    # 訓練並儲存 Model Adam (Categorical output, 正規化資料)
    print("\n訓練 Model Adam...")
    model_adam = Sequential()
    model_adam.add(Input(shape=(11,)))
    model_adam.add(Dense(10, activation="relu"))
    model_adam.add(Dense(8, activation="relu"))
    model_adam.add(Dense(2, activation="softmax"))
    model_adam.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist_adam = model_adam.fit(X_norm, y_cat, epochs=150, batch_size=10, verbose=0)
    loss_adam, acc_adam = model_adam.evaluate(X_norm, y_cat, verbose=0)
    model_adam.save('model_adam.keras')
    print(f"已儲存: model_adam.keras (準確度: {acc_adam:.4f})")
    
    # 訓練並儲存 Model Final (with validation split)
    print("\n訓練 Model Final...")
    model_final = Sequential()
    model_final.add(Input(shape=(11,)))
    model_final.add(Dense(10, activation="relu"))
    model_final.add(Dense(8, activation="relu"))
    model_final.add(Dense(2, activation="softmax"))
    model_final.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist_final = model_final.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=150, batch_size=10, verbose=0)
    loss_tr, acc_tr = model_final.evaluate(X_tr, y_tr, verbose=0)
    loss_te, acc_te = model_final.evaluate(X_te, y_te, verbose=0)
    model_final.save('model_final.keras')
    print(f"已儲存: model_final.keras (訓練: {acc_tr:.4f}, 測試: {acc_te:.4f})")
    
    # 儲存訓練歷史
    training_history = {
        'model_sgd': {
            'accuracy': [float(x) for x in hist_sgd.history['accuracy']],
            'loss': [float(x) for x in hist_sgd.history['loss']]
        },
        'model_adam': {
            'accuracy': [float(x) for x in hist_adam.history['accuracy']],
            'loss': [float(x) for x in hist_adam.history['loss']]
        },
        'model_final': {
            'accuracy': [float(x) for x in hist_final.history['accuracy']],
            'loss': [float(x) for x in hist_final.history['loss']],
            'val_accuracy': [float(x) for x in hist_final.history['val_accuracy']],
            'val_loss': [float(x) for x in hist_final.history['val_loss']]
        }
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f)
    print("已儲存: training_history.json")
    
    # 儲存模型指標
    tf_metrics = {
        'tensorflow': {
            'model_sgd': {'accuracy': float(acc_sgd)},
            'model_adam': {'accuracy': float(acc_adam)},
            'model_final': {
                'train_accuracy': float(acc_tr),
                'test_accuracy': float(acc_te)
            }
        }
    }
    
    with open('tf_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(tf_metrics, f, ensure_ascii=False, indent=2)
    print("已儲存: tf_metrics.json")
    
    print("\n" + "="*50)
    print("所有 TensorFlow 模型已儲存完成！")
    print("="*50)

# 執行儲存（取消下方註解即可執行）
# save_models_for_webapp()