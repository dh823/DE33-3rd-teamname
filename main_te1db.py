import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhive import hive
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="Shopping Data Dashboard", layout="wide")
st.title("🛒 Shopping Data Analysis & Prediction")

@st.cache_data
def load_data():
    conn = hive.Connection(host='localhost', port=10000, username='root', password='root', database='default', auth='LDAP')
    query = '''SELECT country, invoicedate, quantity, unitprice FROM shopping_data'''
    cursor = conn.cursor()
    cursor.execute(query.replace(';', ''))

    try:
        output = cursor.fetchall()
    except:
        output = None

    return output

data = load_data()

if data:
    df = pd.DataFrame(data[1:], columns=["country", "invoicedate", "quantity", "unitprice"])
    df = df[~df["invoicedate"].str.contains("Country", na=False)]
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce", format='%m/%d/%Y %H:%M:%S')
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitprice"] = pd.to_numeric(df["unitprice"], errors="coerce")
    df = df.dropna(subset=["invoicedate", "quantity", "unitprice"])
    df["TotalRevenue"] = df["quantity"] * df["unitprice"]

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    df_country_summary = df.groupby("country")[["quantity", "unitprice"]].mean().reset_index()

    st.subheader("🌍 Average Quantity & Unit Price by Country")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_country_summary, x="quantity", y="country", palette="coolwarm", ax=ax)
    plt.xlabel("Average Quantity")
    plt.ylabel("Country")
    plt.title("Average Quantity by Country")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_country_summary, x="unitprice", y="country", palette="coolwarm", ax=ax)
    plt.xlabel("Average Unit Price")
    plt.ylabel("Country")
    plt.title("Average Unit Price by Country")
    st.pyplot(fig)

    df_revenue = df.groupby("invoicedate")["TotalRevenue"].sum().reset_index()
    df_revenue["invoicedate_numeric"] = df_revenue["invoicedate"].map(datetime.toordinal)

    X = df_revenue[["invoicedate_numeric"]]
    Y = df_revenue["TotalRevenue"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    st.subheader("📈 Revenue Prediction using Linear Regression")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X_test, Y_test, color="blue", label="Actual Revenue")
    ax.plot(X_test, Y_pred, color="red", label="Predicted Revenue")
    plt.xlabel("Date")
    plt.ylabel("Total Revenue")
    plt.title("Revenue Prediction")
    plt.legend()
    st.pyplot(fig)

    st.subheader("📉 Total Revenue by Date")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_revenue, x="invoicedate", y="TotalRevenue", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.title("Total Revenue by Date")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)
