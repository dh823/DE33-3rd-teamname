import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhive import hive
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 스트림릿 페이지 설정
st.set_page_config(page_title="Sales Prediction", page_icon="📈")
st.sidebar.header("Select Option")
option = st.sidebar.radio(
    "Choose a view", ["고객별 매출", "월별 매출"]
)


# 데이터 캐싱 및 연결
@st.cache_data
def load_data():
    conn = hive.Connection(host='localhost', port = 10000, username = 'root', password = 'root', database = 'default', auth = 'LDAP')

    query = '''SELECT * FROM shopping_data;'''

    cursor = conn.cursor()
    cursor.execute(query.replace(';',''))
    #df = pd.read_sql(query, conn)

    try:
        output = cursor.fetchall()
    except:
        output = None

    columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice','CustomerID','Country']
    df = pd.DataFrame(output, columns = columns)
    
    # 첫 번째 행이 컬럼명이기 때문에, 첫 번째 행 제거
    df = df.iloc[1:].reset_index(drop=True)
    
    # 데이터 전처리
    # 날짜 데이터 변환
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors = 'coerce')
    # 결측치 처리
    df = df.dropna(subset = ['Quantity'])
    df = df.dropna(subset = ['Description'])
    # UnitPrice 음수 제거
    df = df[df['UnitPrice'] >= 0]
    # 날짜 정보 추출
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    # 총 매출
    df['TotalRevenue'] = df['Quantity']*df['UnitPrice']
    return df

df = load_data()

# 고객별 매출 분석
if option == '고객별 매출':
    st.title('Customer Revenue Analysis')

    # 고객별 총 매출액 Top5
    customer_revenue = df.groupby(['CustomerID', 'Country'])['TotalRevenue'].sum().reset_index()
    customer_revenue = customer_revenue.sort_values(by='TotalRevenue', ascending=False)

    # 고객별 구매 횟수 Top5
    customer_transactions = df.groupby(['CustomerID', 'Country'])['InvoiceNo'].nunique().reset_index()
    customer_transactions = customer_transactions.sort_values(by='InvoiceNo', ascending=False)

    # 고객별 월별 매출 
    customer_monthly_revenue = df.groupby(['CustomerID', 'Country', 'Year', 'Month'])['TotalRevenue'].sum().reset_index()

    # 고객별 최근 구매일
    customer_recent_purchase = df.groupby(['CustomerID', 'Country'])['InvoiceDate'].max().reset_index()

    # 2개의 표를 한 줄에 배치
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Customers by Revenue")
        st.write(customer_revenue.head())

    with col2:
        st.subheader("Top 5 Customers by Number of Transactions")
        st.write(customer_transactions.head())

    # 두 번째 줄에서 또 다른 표 2개를 나란히 배치
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Customer Monthly Revenue")
        st.write(customer_monthly_revenue.head())

    with col4:
        st.subheader("Customer Recent Purchase Date")
        st.write(customer_recent_purchase.head())





# 월별 매출 분석
elif option == '월별 매출':
    st.title('Monthly Revenue Analysis')

    # 'YearMonth'를 Period 형식에서 datetime으로 변환
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_revenue = df.groupby('YearMonth')['TotalRevenue'].sum().reset_index()

    # ARIMA 모델 학습 (최적 파라미터 (p,d,q)는 데이터에 맞게 튜닝 필요)
    arima_model = ARIMA(monthly_revenue['TotalRevenue'], order=(5, 1, 0))  # (p,d,q)값은 데이터에 맞게 최적화
    arima_model_fit = arima_model.fit()

    # 예측 수행 (다음 3개월 예측) - ARIMA
    forecast_steps = 3
    forecast_arima = arima_model_fit.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(monthly_revenue['YearMonth'].iloc[-1].end_time, periods=forecast_steps + 1, freq='M')[1:]

    # Linear Regression 예측 (선형 회귀)
    X = np.array(range(len(monthly_revenue))).reshape(-1, 1)  # 연속적인 숫자로 X 값 설정
    y = monthly_revenue['TotalRevenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    forecast_lr = lr_model.predict(np.array(range(len(monthly_revenue), len(monthly_revenue) + forecast_steps)).reshape(-1, 1))

    # KNN 모델의 예측을 위한 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # KNN 모델 학습 및 예측
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    forecast_knn = knn_model.predict(scaler.transform(np.array(range(len(monthly_revenue), len(monthly_revenue) + forecast_steps)).reshape(-1, 1)))

    # 실제 데이터의 날짜를 datetime 형식으로 변환
    actual_dates = pd.to_datetime(monthly_revenue['YearMonth'].astype(str))

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))

    # 실제 매출 (Actual Revenue) 그리기
    ax.plot(actual_dates, monthly_revenue['TotalRevenue'], label='Actual Revenue', color='blue')

    # ARIMA 예측 (ARIMA Forecast) 그리기
    ax.plot(forecast_dates, forecast_arima, label='ARIMA Forecast', color='red')

    # 선형 회귀 예측 (Linear Regression Forecast) 그리기
    ax.plot(forecast_dates, forecast_lr, label='Linear Regression Forecast', color='green')

    # KNN 예측 (KNN Forecast) 그리기
    ax.plot(forecast_dates, forecast_knn, label='KNN Forecast', color='orange')

    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.set_title(f"Revenue Forecast Comparison: ARIMA vs Linear Regression vs KNN")
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)  # x축 레이블 회전

    st.pyplot(fig)
