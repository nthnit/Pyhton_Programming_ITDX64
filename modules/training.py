import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def show(data, tr):
    st.header(tr("section3_header"))
    
    if "quality" in data.columns:
        X = data.drop("quality", axis=1)
        y = data["quality"]
    else:
        st.error(tr("error_quality_missing"))
        st.stop()
    
    # Choose test size
    test_size_percent = st.slider(tr("section3_test_size"), min_value=10, max_value=50, value=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_percent/100, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    model_choice = st.selectbox(
        tr("section3_choose_model"),
        ["KNN", "Decision Tree", "SVM", "Logistic Regression"]
    )

    if model_choice == "KNN":
        k = st.slider(tr("section3_k"), min_value=1, max_value=20, value=5, step=1)
        model = KNeighborsClassifier(n_neighbors=k)

    elif model_choice == "Decision Tree":
        max_depth = st.slider(tr("section3_dt_max_depth"), min_value=1, max_value=20, value=5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    elif model_choice == "SVM":
        c_value = st.slider(tr("section3_svm_c"), min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        model = SVC(C=c_value, kernel='rbf', random_state=42)

    elif model_choice == "Logistic Regression":
        c_value = st.slider(tr("section3_lr_c"), min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Evaluation
    st.subheader(tr("section3_model_eval"))
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Save to session state
    st.session_state.scaler = scaler
    st.session_state.model = model
