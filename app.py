import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('gpascore.csv')

# 데이터 전처리(결측치 처리)
# print(data.isnull().sum()) 빈칸의 개수를 보여줌
data = data.dropna()  # 빈값이 있는 행을 제거
# print(data.isnull().sum())
# data = data.fillna(100) 빈칸을 내가 원하는 값으로 채워줌

x = []
y = data['admit'].values

# data라는 데이터 프레임을 가로 한 줄씩 출력해 주라
for i, rows in data.iterrows():
    x.append([rows['gre'], rows['gpa'], rows['rank']])
    
print(x)
    

model = tf.keras.models.Sequential([
    # 숫자 같은 경우 계속 돌려 보면서 정확도가 높은 방향으로 모델을 튜닝시켜 주는 작업이 필요함
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    # 마지막 레이어는 예측결과(sigmoid -> 0~1확률)
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습 데이터, 실제 정답
model.fit(np.array(x), np.array(y), epochs=1000)

# 예측
predict = model.predict([ [750, 3.70, 3], [400, 2.2, 1] ])
print(predict)


