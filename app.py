from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

model = load_model('model/best_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    #데이터 받을 빈 리스트 생성
    data_list = []  

    #사용자가 입력한 값 불러오기
    data_list = request.get_json()['data_list']
    print(data_list)

    #데이터 형태 맞춰주기
    input_query = np.array([data_list], dtype=np.float32)
    print(input_query)

	#모델과 비교
    result = model.predict(input_query)
    
    # 백분위로 변환하고 소수점 뒤에 숫자 없애기
    result = int(result[0] * 100)

    #json 형식으로 return
    return jsonify({'합격 확률' : f'{result}%'})


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, port=8080)