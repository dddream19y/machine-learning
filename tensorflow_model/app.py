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