import streamlit as st

def show(translations, tr):
    st.title(tr("title"))
    st.header(tr("section1_header"))
    st.markdown(tr("section1_body"))
