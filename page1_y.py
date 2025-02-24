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
SELECT Country, Quantity
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
df = pd.DataFrame(output, columns=["Country", "Quantity"])

# 국가별 총 주문량 계산
country_sales = df.groupby("Country")["Quantity"].sum().reset_index()

# 매출 상위 5개국 추출
top5_countries = country_sales.sort_values(by="Quantity", ascending=False).head(5)

# 영국 제외
df2 = df[df["Country"] != "United Kingdom"]

# 국가별 총 주문량 계산
country_sales2 = df2.groupby("Country")["Quantity"].sum().reset_index()

# 매출 상위 5개국 추출
top5_countries2 = country_sales.sort_values(by="Quantity", ascending=False).head(5)

# page configuration
st.set_page_config(
    page_title="E-commerce Dashboard",
    page_icon="💲 ",
    layout="wide",
    initial_sidebar_state="expanded")

st.title('📊  TOP 5 판매 국가')

# tab
tab1, tab2 = st.tabs(["📈 TOP 5", "📈 영국 제외 TOP 5"])

# 바그래프 생성 함수
def plot_top5(top5_countries):
    # 시각화 (막대 그래프)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top5_countries, x="Quantity", y="Country", palette="coolwarm")

    # 그래프 스타일 설정
    plt.xlabel("Total Quantity")
    plt.ylabel("Country")
#    plt.title("Top 5 Countries by Total Quantity")
    plt.grid(axis='x', linestyle="--")
    plt.show()

    return plt

# 영국 제외 그래프 생성 함수
def plot_top5_uk(top5_countries2):
    # 시각화 (막대 그래프)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top5_countries2, x="Quantity", y="Country", palette="coolwarm")

    # 그래프 스타일 설정
    plt.xlabel("Total Quantity")
    plt.ylabel("Country")
#    plt.title("Top 5 Countries without UK by Total Quantity")
    plt.grid(axis='x', linestyle="--")
    plt.show()

    return plt

with tab1:
#    img = Image.open('top5_countries.png')
#    st.image(img)
    fig = plot_top5(top5_countries)  # 그래프 생성
    st.pyplot(fig)  # Streamlit에서 Matplotlib 그래프 렌더링

with tab2:
#    img = Image.open('top5_countries_uk.png')
#    st.image(img)

    fig = plot_top5_uk(top5_countries2)  # 그래프 생성
    st.pyplot(fig)  # Streamlit에서 Matplotlib 그래프 렌더링


