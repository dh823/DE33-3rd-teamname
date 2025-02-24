import streamlit as st

pages = {
        "통계": [
                st.Page("page1.py", title="TOP5 판매국가"),
            ],    
        "예측": [
                st.Page("page2.py", title="선형회귀"), 
                st.Page("page3.py",title="ARIMA")
            ]
        }

pg = st.navigation(pages)
pg.run()
