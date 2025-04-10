import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show(data, tr):
    st.header(tr("section2_header"))
    st.subheader(tr("section2_overview"))
    st.write(tr("section2_shape"), data.shape)
    st.write(tr("section2_columns"), list(data.columns))
    if st.checkbox(tr("section2_head")):
        st.write(data.head())
    
    # Gộp cả lựa chọn trực quan (cũ và mới) vào cùng một radio
    st.sidebar.header(tr("section2_sidebar_header"))
    eda_options = (
        tr("section2_sidebar_option1"),  # Hiển thị bảng dữ liệu
        tr("section2_sidebar_option2"),  # Biểu đồ phân phối
        tr("section2_sidebar_option3"),  # Heatmap tương quan
        "Box Plot",
        "Count Plot",
        "Pair Plot",
        "Violin Plot",
        "KDE Plot"
    )
    eda_option = st.sidebar.radio("Choose Visualization:", eda_options)
    
    if eda_option == tr("section2_sidebar_option1"):
        st.subheader(tr("section2_sidebar_option1"))
        st.write(data.head())
        
    elif eda_option == tr("section2_sidebar_option2"):
        st.subheader(tr("section2_sidebar_option2"))
        col_to_plot = st.selectbox("Choose a column:", data.columns)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(data[col_to_plot].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {col_to_plot}")
        ax.set_xlabel(col_to_plot)
        ax.set_ylabel("Frequency")
        st.pyplot(fig, use_container_width=False)
        
    elif eda_option == tr("section2_sidebar_option3"):
        st.subheader(tr("section2_sidebar_option3"))
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=False)
        
    elif eda_option == "Box Plot":
        st.subheader("Box Plot")
        box_feature = st.selectbox("Choose feature for Box Plot:", data.columns)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=data[box_feature], ax=ax, color="lightblue")
        ax.set_title(f"Box Plot for {box_feature}")
        st.pyplot(fig, use_container_width=False)
        
    elif eda_option == "Count Plot":
        st.subheader("Count Plot")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=data["quality"], ax=ax, palette="viridis")
        ax.set_title("Count Plot of Wine Quality")
        st.pyplot(fig, use_container_width=False)
        
    elif eda_option == "Pair Plot":
        st.subheader("Pair Plot")
        selected_features = st.multiselect("Choose features for Pair Plot:", list(data.columns), default=list(data.columns)[:4])
        if len(selected_features) >= 2:
            pairgrid = sns.pairplot(data[selected_features])
            st.pyplot(pairgrid.fig, use_container_width=True)
        else:
            st.info("Please select at least 2 features for Pair Plot.")
            
    elif eda_option == "Violin Plot":
        st.subheader("Violin Plot")
        feature = st.selectbox("Choose a feature for Violin Plot:", [col for col in data.columns if col != "quality"])
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.violinplot(x="quality", y=feature, data=data, ax=ax, palette="muted")
        ax.set_title(f"Violin Plot of {feature} by Wine Quality")
        st.pyplot(fig, use_container_width=False)
        
    elif eda_option == "KDE Plot":
        st.subheader("KDE Plot")
        feature = st.selectbox("Choose a feature for KDE Plot:", data.columns)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data[feature].dropna(), ax=ax, shade=True, color="green")
        ax.set_title(f"KDE Plot for {feature}")
        st.pyplot(fig, use_container_width=False)
