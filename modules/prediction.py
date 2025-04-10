import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show(data, tr):
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: rgb(0, 102, 204) !important;
            color: white !important;
            border: none !important;
        }
        div.stButton > button:hover {
            background-color: rgb(0, 82, 184) !important;
        }
        .focused {
            border-color: rgb(0, 102, 204) !important;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.header(tr("section4_header"))
    if "knn" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Please go to Section 3: Preprocessing & Training first.")
    else:
        scaler = st.session_state.scaler
        knn = st.session_state.knn
        
        st.markdown(tr("section4_description"))
        fixed_acidity = st.number_input("Fixed Acidity", value=float(data["fixed acidity"].mean()))
        volatile_acidity = st.number_input("Volatile Acidity", value=float(data["volatile acidity"].mean()))
        citric_acid = st.number_input("Citric Acid", value=float(data["citric acid"].mean()))
        residual_sugar = st.number_input("Residual Sugar", value=float(data["residual sugar"].mean()))
        chlorides = st.number_input("Chlorides", value=float(data["chlorides"].mean()))
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=float(data["free sulfur dioxide"].mean()))
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=float(data["total sulfur dioxide"].mean()))
        density = st.number_input("Density", value=float(data["density"].mean()))
        pH = st.number_input("pH", value=float(data["pH"].mean()))
        sulphates = st.number_input("Sulphates", value=float(data["sulphates"].mean()))
        alcohol = st.number_input("Alcohol", value=float(data["alcohol"].mean()))
        
        if st.button(tr("predict_button")):
            # Lấy tên các feature từ dữ liệu gốc (loại bỏ cột 'quality')
            feature_names = list(data.drop("quality", axis=1).columns)
            # Tạo DataFrame từ các giá trị input với các tên cột phù hợp
            input_features = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                             free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                                          columns=feature_names)
            input_scaled = scaler.transform(input_features)
            prediction = knn.predict(input_scaled)
            st.success(f"Predicted Wine Quality: {prediction[0]}")
            
            # Minh họa kết quả bằng biểu đồ xác suất của các lớp
            proba = knn.predict_proba(input_scaled)[0]
            classes = knn.classes_
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(classes, proba, color='skyblue', edgecolor='black')
            ax.set_xlabel("Wine Quality")
            ax.set_ylabel("Probability")
            ax.set_title("Predicted Probability Distribution")
            # Hiển thị giá trị xác suất trên từng cột
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f"{height:.2f}", ha='center', va='bottom')
            st.pyplot(fig, use_container_width=False)
