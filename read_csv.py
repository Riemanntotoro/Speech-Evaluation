import pandas as pd

# 모든 행·열을 다 보이도록 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("denoised.csv")
print(df)
