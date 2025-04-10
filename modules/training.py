import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def show(data, tr):
    st.header(tr("section3_header"))
    if "quality" in data.columns:
        X = data.drop("quality", axis=1)
        y = data["quality"]
    else:
        st.error(tr("error_quality_missing"))
        st.stop()
    
    test_size_percent = st.slider(tr("section3_test_size"), min_value=10, max_value=50, value=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent/100, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k = st.slider(tr("section3_k"), min_value=1, max_value=20, value=5, step=1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader(tr("section3_model_eval"))
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))
    
    # Lưu đối tượng vào session_state để sử dụng sau này
    st.session_state.scaler = scaler
    st.session_state.knn = knn
