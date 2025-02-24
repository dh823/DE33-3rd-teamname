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
df = df[df["Country"] != "United Kingdom"]

# 국가별 총 판매량 계산
country_sales = df.groupby("Country")["Quantity"].sum().reset_index()
top_5_countries = country_sales.sort_values(by="Quantity", ascending=False).head(5)["Country"].tolist()

# MSE 저장
mse_results = {}

# page configuration
st.set_page_config(
    page_title="E-commerce Dashboard",
    page_icon="💲 ",
    layout="wide",
    initial_sidebar_state="expanded")

st.header("📈 판매량 TOP5 국가 예측")
st.subheader("- 영국 제외")

#st.title('📊 선형회귀 예측')


# 바그래프 생성 함수
# 그래프 그리기

def plot_top5(top_5_countries):
    plt.figure(figsize=(12, 6))

    for country in top_5_countries:
        # 특정 국가 필터링
        country_df = df[df["Country"] == country]

        # 월별 주문량 계산
        country_df["Month"] = country_df["InvoiceDate"].dt.to_period("M")
        monthly_orders = country_df.groupby("Month")["Quantity"].sum().reset_index()

        # 'Month'를 숫자로 변환 (예측을 위해)
        monthly_orders["Month"] = monthly_orders["Month"].astype(str)
        monthly_orders["Month_Num"] = range(1, len(monthly_orders) + 1)

        # 선형회귀 모델 학습
        X = monthly_orders[["Month_Num"]]
        y = monthly_orders["Quantity"]
        model = LinearRegression()
        model.fit(X, y)

        # 예측 및 MSE 계산 (훈련 데이터셋에서)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mse_results[country] = mse
    
        # 미래 6개월 예측
        last_date = datetime.strptime(monthly_orders["Month"].max(), "%Y-%m")
        future_dates = [(last_date + timedelta(days=30 * i)).strftime("%Y-%m") for i in range(1, 7)]
        future_months = pd.DataFrame({"Month": future_dates, "Month_Num": range(X["Month_Num"].max() + 1, X["Month_Num"].max() + 7)})
        future_predictions = model.predict(future_months[["Month_Num"]])

        # 기존 데이터 및 예측 데이터 시각화
        plt.plot(monthly_orders["Month"], monthly_orders["Quantity"], label=f"{country} (Actual)")
        plt.plot(future_months["Month"], future_predictions, linestyle="dashed", label=f"{country} (Predicted)")

    plt.xlabel("Date")
    plt.ylabel("Total Quantity")
    # plt.title("6-Month Order Prediction for Top 5 Countries (Linear Regression)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    return plt

# MSE 결과 출력
print("각 국가별 MSE(Mean Squared Error) 결과:")
for country, mse in mse_results.items():
    print(f"{country}: {mse:.2f}")


#img = Image.open('lr_top5_month.png')
#st.header("📈 판매량 TOP5 국가 예측")
#st.subheader("- 영국 제외")
#st.image(img)
fig = plot_top5(top5_countries)  # 그래프 생성
st.pyplot(fig)  # Streamlit에서 Matplotlib 그래프 렌더링

