from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np

model = load_model('model/university_best_model.h5')

# Tokenizer : 문자열 -> 숫자 데이터로 임베딩 
tokenizer = Tokenizer(filters='')

data_list = ['KAIST', '전학부', '종합', 1, 1, 1, 300, 1]  

input_sequence = tokenizer.texts_to_sequences([data_list[0:3]])
input = np.concatenate((input_sequence, [data_list[3:]]), axis=1, dtype=np.float32)

print(input)

input_data = input.tolist()

#모델과 비교
result = model.predict(input_data)

# 백분위로 변환하고 소수점 뒤에 숫자 없애기
result = int(result[0] * 100)

#json 형식으로 return   
print(result)