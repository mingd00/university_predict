import pandas as pd
import json

with open('json/2019_pass_or_fail.json', encoding='UTF8') as file:
    dataset = json.load(file)
    
# JSON 데이터로부터 DataFrame 생성
df = pd.DataFrame(dataset)

# 필요한 필드(지역, 대학명, 계열, 학과) 추출 및 중복 제거
df_filtered = df[['지역', '대학명', '계열', '학과명']].drop_duplicates()

print(df_filtered[0:5])

# 결과를 CSV 파일로 저장
df_filtered.to_csv('universityData.csv', index=False)