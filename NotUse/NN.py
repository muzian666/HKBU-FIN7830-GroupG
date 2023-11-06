import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 加载和准备数据
df = pd.read_csv('../Resources/Data/Encoded_Resampled_HR_Analytics.csv')
# X = df.drop(['Attrition', 'Over18', 'StandardHours', 'Department'], axis=1)
X = df[['YearsWithCurrManager', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'TrainingTimesLastYear', 'Age',
        'JobInvolvement', 'MaritalStatus']]
y = df['Attrition']
# y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建模型
model = Sequential([
    # 输入层
    Dense(32, input_shape=(X_train_scaled.shape[1],)),  # 减少神经元数量
    LeakyReLU(alpha=0.3),
    Dropout(0.2),

    # 隐藏层1
    Dense(16, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)),  # 增加正则化
    LeakyReLU(alpha=0.3),
    Dropout(0.4),  # 调整Dropout比率

    # 隐藏层2（删除，以减少模型复杂性）

    # 输出层
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2000, verbose=1, mode='auto', baseline=0.95, restore_best_weights=True)

# 训练模型
model.fit(X_train_scaled, y_train, epochs=2000, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback, early_stopping])

# 评估模型
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# 可视化
tf.keras.utils.plot_model(model, to_file='../model.png', show_shapes=True)
tf.saved_model.save(model, '../Resources/Model/Model')
