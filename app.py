import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality Prediction", page_icon="assets/logoFIT.svg", layout="wide")
st.markdown("""
<style>
	[data-testid="stDecoration"] {
		background-image: linear-gradient(90deg, rgb(0, 102, 204), rgb(102, 255, 255));
	}
    .st-d3 {
    background-color: rgb(34, 83, 192);
    }
}


 
</style>""",
unsafe_allow_html=True)

st.sidebar.image("assets/logoFIT.svg", width=150)
# ----------------- ĐỌC FILE TRANSLATIONS -----------------
@st.cache_data
def load_translations():
    with open("lang/translations.json", "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()

# ----------------- CHỌN NGÔN NGỮ -----------------

selected_lang_display = st.sidebar.selectbox("Select Language / Chọn Ngôn ngữ", ["English", "Tiếng Việt"])
language_mapping = {"English": "en", "Tiếng Việt": "vi"}
selected_language = language_mapping[selected_lang_display]

def tr(key):
    return translations[selected_language].get(key, key)

# ----------------- LOAD DỮ LIỆU -----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Data/winequality-red.csv", sep=";")
    return data

data = load_data()

# ----------------- IMPORT MODULES -----------------
from modules import introduction, eda, training, prediction

# ----------------- NAVIGATION -----------------

nav = st.sidebar.radio("Navigation", [tr("nav1"), tr("nav2"), tr("nav3"), tr("nav4")])

if nav == tr("nav1"):
    introduction.show(translations, tr)
elif nav == tr("nav2"):
    eda.show(data, tr)
elif nav == tr("nav3"):
    training.show(data, tr)
elif nav == tr("nav4"):
    prediction.show(data, tr)
