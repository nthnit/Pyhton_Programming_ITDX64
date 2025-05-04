import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show(data, tr):
    # Custom CSS for buttons (same as before)
    st.markdown(
    """
    <style>
    section.main div.stButton > button {
        background-color: rgb(0, 102, 204) !important;
        color: white !important;
        border: none !important;
    }
    section.main div.stButton > button:hover {
        background-color: rgb(0, 82, 184) !important;
    }
    section.main .focused {
        border-color: rgb(0, 102, 204) !important;
    }
    </style>
    """, unsafe_allow_html=True
    )

    st.header(tr("section4_header"))
    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Please go to Section 3: Preprocessing & Training first.")
        return

    scaler = st.session_state.scaler
    model  = st.session_state.model
    feature_names = list(data.drop("quality", axis=1).columns)

    st.markdown(tr("section4_description"))

    # --- SINGLE SAMPLE INPUT --------------------------------------------------
    st.subheader("Single-sample Prediction")
    with st.expander("Single-sample prediction", expanded=True):
        inputs = {
            "fixed acidity": st.number_input("Fixed Acidity", value=float(data["fixed acidity"].mean())),
            "volatile acidity": st.number_input("Volatile Acidity", value=float(data["volatile acidity"].mean())),
            "citric acid": st.number_input("Citric Acid", value=float(data["citric acid"].mean())),
            "residual sugar": st.number_input("Residual Sugar", value=float(data["residual sugar"].mean())),
            "chlorides": st.number_input("Chlorides", value=float(data["chlorides"].mean())),
            "free sulfur dioxide": st.number_input("Free Sulfur Dioxide", value=float(data["free sulfur dioxide"].mean())),
            "total sulfur dioxide": st.number_input("Total Sulfur Dioxide", value=float(data["total sulfur dioxide"].mean())),
            "density": st.number_input("Density", value=float(data["density"].mean())),
            "pH": st.number_input("pH", value=float(data["pH"].mean())),
            "sulphates": st.number_input("Sulphates", value=float(data["sulphates"].mean())),
            "alcohol": st.number_input("Alcohol", value=float(data["alcohol"].mean()))
        }
        if st.button(tr("predict_button")):
            df_single = pd.DataFrame([inputs], columns=feature_names)
            Xs = scaler.transform(df_single)
            y_pred = model.predict(Xs)
            st.success(f"Predicted Wine Quality: {y_pred[0]}")
            
            # show probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xs)[0]
                classes = model.classes_
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.bar(classes, proba, color='skyblue', edgecolor='black')
                ax.set_xlabel("Quality")
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probabilities")
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2, h, f"{h:.2f}", ha='center', va='bottom')
                st.pyplot(fig, use_container_width=False)

    # --- BATCH PREDICTION -----------------------------------------------------
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV of samples", type="csv", help="CSV must contain exactly these columns: " + ", ".join(feature_names))
    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            # Validate columns
            missing = set(feature_names) - set(df_batch.columns)
            extra   = set(df_batch.columns)  - set(feature_names)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            elif extra:
                st.warning(f"Ignoring extra columns: {', '.join(extra)}")
                df_batch = df_batch[feature_names]
            # Scale & predict
            Xb = scaler.transform(df_batch)
            preds = model.predict(Xb)
            df_batch["predicted_quality"] = preds
            # Probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xb)
                for idx, cls in enumerate(model.classes_):
                    df_batch[f"prob_{cls}"] = proba[:, idx]
            st.dataframe(df_batch)
            # Download button
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="wine_quality_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
