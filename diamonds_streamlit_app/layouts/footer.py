import streamlit as st
import datetime


def footer():
    st.html("<hr>")
    date = datetime.date.today().year
    st.markdown(f"coding_week @ {date}_")