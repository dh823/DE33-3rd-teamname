import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhive import hive
from datetime import datetime

import streamlit as st

from PIL import Image

# Hive 연결 설정
conn = hive.Connection(
    host='localhost', 
    port=10000, 
    username='root', 
    password='root', 
    database='default', 
    auth='LDAP'
)

# Hive에서 데이터 가져오기
query = """
SELECT InvoiceDate, Country, Quantity
FROM online_retail_data
"""

cursor = conn.cursor()
cursor.execute(query)

# 데이터를 Pandas DataFrame으로 변환
try:
    output = cursor.fetchall()
except:
    output = None

# 컬럼명 설정 (테이블의 실제 스키마에 맞게 조정)
df = pd.DataFrame(output, columns=["InvoiceDate", "Country", "Quantity"])

# 날짜 변환
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 영국 제외
df = df[df['Country'] != 'United Kingdom']

# 국가별 총 주문량 계산 후 상위 5개국 선택
top_countries = df.groupby("Country")["Quantity"].sum().nlargest(5).index.tolist()
df = df[df['Country'].isin(top_countries)]

# 날짜를 월별로 변환
df['Month'] = df['InvoiceDate'].dt.to_period('M')

# 월별 주문량 집계
df_grouped = df.groupby(['Month', 'Country'])['Quantity'].sum().reset_index()

# ARIMA 예측 및 시각화
future_months = 6
mse_results = {}

# page configuration
st.set_page_config(
    page_title="E-commerce Dashboard",
    page_icon="💲 ",
    layout="wide",
    initial_sidebar_state="expanded")

#st.title('📊 ARIMA 예측')
st.header("📈 판매량 TOP5 국가 예측")
st.subheader("- 영국 제외")


# 바그래프 생성 함수
def plot_top5(top5_countries):
    # 그래프 초기화
    plt.figure(figsize=(15, 8))

    for country in top_countries:
        country_data = df_grouped[df_grouped['Country'] == country]

        # 월을 인덱스로 설정 (datetime으로 변환)
        country_data['Month'] = country_data['Month'].astype(str)  # 'YYYY-MM' 형식
        country_data['Month'] = pd.to_datetime(country_data['Month'])
        country_data = country_data.set_index('Month')

        # 학습 데이터 정의 (80% 학습, 20% 테스트)
        train_size = int(len(country_data) * 0.8)
        train_data, test_data = country_data[:train_size], country_data[train_size:]

        try:
            # ARIMA 모델 학습
            model = ARIMA(train_data['Quantity'], order=(1, 1, 1))
            model_fit = model.fit()

            # 미래 6개월 예측
            future_forecast = model_fit.forecast(steps=future_months)

            # 예측 날짜 생성
            future_dates = pd.date_range(start=country_data.index[-1], periods=future_months + 1, freq='M')[1:]
            
            # MSE 계산 (테스트 데이터가 있을 경우)
            if len(test_data) > 0:
                predicted_values = model_fit.forecast(steps=len(test_data))
                mse = mean_squared_error(test_data['Quantity'], predicted_values)
                mse_results[country] = mse

            # 그래프 그리기
            plt.plot(country_data.index, country_data['Quantity'], linestyle="-", label=f"Actual - {country}")
            plt.plot(future_dates, future_forecast, linestyle="dashed", label=f"Predicted - {country}")

        except Exception as e:
            print(f"ARIMA 모델 학습 실패 - {country}: {e}")

    plt.xlabel("Month")
    plt.ylabel("Total Quantity")
    # plt.title("Future Order Prediction (Top 5 Countries, Excluding UK) - ARIMA (Next 6 Months)")
    plt.legend()
    plt.grid()
    plt.show()

    return plt

#img = Image.open('arima_top5.png')

fig = plot_top5(top5_countries)  # 그래프 생성
st.pyplot(fig)  # Streamlit에서 Matplotlib 그래프 렌더링


