import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def show(data, tr):
    st.header(tr("section3_header"))
    if "quality" not in data.columns:
        st.error(tr("error_quality_missing"))
        return

    X = data.drop("quality", axis=1)
    y = data["quality"]

    # 1. Train/test split
    test_size_percent = st.slider(tr("section3_test_size"), 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_percent/100, random_state=42
    )

    # 2. Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. Model selection
    model_name = st.selectbox(
        tr("section3_choose_model"),
        ["KNN", "Decision Tree", "SVM", "Logistic Regression"]
    )

    # 4. Hyperparameter tuning toggle
    do_tune = st.checkbox("Enable Hyperparameter Tuning (GridSearchCV)", value=False)

    if model_name == "KNN":
        base = KNeighborsClassifier()
        param_grid = {"n_neighbors": list(range(1, 21))}
    elif model_name == "Decision Tree":
        base = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": list(range(1, 21))}
    elif model_name == "SVM":
        base = SVC(kernel="rbf", probability=True, random_state=42)
        param_grid = {"C": [0.01, 0.1, 1, 10], "gamma": ["scale", "auto"]}
    else:  # Logistic Regression
        base = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {"C": [0.01, 0.1, 1, 10]}

    # 5. Either tune or use default sliders
    if do_tune:
        st.write("Running GridSearchCV... this may take a moment.")
        grid = GridSearchCV(base, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_
        st.write(f"Best parameters: {grid.best_params_}")
        st.write(f"CV Accuracy: {grid.best_score_:.3f}")
    else:
        # fallback: manual sliders for a single hyperparameter if desired
        if model_name == "KNN":
            k = st.slider(tr("section3_k"), 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_name == "Decision Tree":
            max_depth = st.slider(tr("section3_dt_max_depth"), 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_name == "SVM":
            c = st.slider(tr("section3_svm_c"), 0.01, 10.0, 1.0, 0.01)
            model = SVC(C=c, kernel="rbf", probability=True, random_state=42)
        else:
            c = st.slider(tr("section3_lr_c"), 0.01, 10.0, 1.0, 0.01)
            model = LogisticRegression(C=c, max_iter=1000, random_state=42)

        # train the chosen model
        model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader(tr("section3_model_eval"))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    
    # — Confusion Matrix Heatmap —
    st.write("Confusion Matrix:")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=model.classes_, yticklabels=model.classes_)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    st.pyplot(fig_cm, use_container_width=False)
    
    # — Classification Report —
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))


    # 7. Persist
    st.session_state.scaler = scaler
    st.session_state.model  = model
