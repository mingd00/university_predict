from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from flask import jsonify
import numpy as np
import pandas as pd
import json

model = load_model('model/university_best_model.h5')

def create_tokenizer():
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
    
    data_list = ['DGIST', '융복합대학(기초학부)', '종합', 1, 1, 1, 300, 1]  

    input_sequence = tokenizer.texts_to_sequences([data_list[0:3]])
    
    input = np.concatenate((input_sequence, [data_list[3:]]), axis=1, )
    
    input_data = input.tolist()
    
    predict = model.predict(input_data)
    
    # 백분위로 변환하고 소수점 뒤에 숫자 없애기
    result = int(predict[0] * 100)

    #json 형식으로 return   
    return result


if __name__ == "__main__":
    print(create_tokenizer())
