from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import json

model = load_model('model/university_best_model.h5')

with open('json/2019_pass_or_fail.json', encoding='UTF8') as file:
    dataset = json.load(file)
    
# JSON 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(dataset)

# "최저 지원 유무" 열 삭제
df_modify = df.drop(columns=['최저적용유무', '최초지원결과', '지역', '전형명', '계열'])

# 빈 값이 있는 행 제거
df_modify = df_modify.replace('', pd.NA).dropna()

# 결과를 다시 JSON 형식으로 변환
modified_data = df_modify.to_dict(orient='records')
    
# 문자열 데이터를 numpy 배열로 변환 
string_data = np.array([
    [
        data['대학명'],
        data['학과명'],
        data['전형유형'],
    ]
    for data in modified_data
])

# 각 행을 문자열로 합치기
texts = [' '.join(row) for row in string_data]

# Tokenizer : 문자열 -> 숫자 데이터로 임베딩 
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(texts)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict(): 
    # tokenizer를 글로벌 변수로 선언
    global tokenizer  
    
    #데이터 받을 빈 리스트 생성
    data_list = []  

    #사용자가 입력한 값 불러오기
    data_list = request.get_json()['data_list']
    
    input_sequence = tokenizer.texts_to_sequences([data_list[0:3]])
    input = np.concatenate((input_sequence, [data_list[3:]]), axis=1, dtype=np.float32)
    
    input_data = input.tolist()

	#모델과 비교
    result = model.predict(input_data)
    
    # 백분위로 변환하고 소수점 뒤에 숫자 없애기
    result = int(result[0] * 100)
    
    #json 형식으로 return
    return jsonify({'합격 확률' : f'{result}%'})


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, port=7070)