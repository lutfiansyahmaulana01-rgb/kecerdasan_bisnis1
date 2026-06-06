import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
rf_model = joblib.load("model_rf_dbd.joblib")
knn_model = joblib.load("model_knn_dbd.joblib")
svm_model = joblib.load("model_svm_dbd.joblib")

# =========================
# JUDUL
# =========================
st.set_page_config(page_title="Prediksi DBD", layout="wide")

st.title("🩺 Sistem Prediksi DBD")
st.write("Prediksi DBD menggunakan model Random Forest, KNN, dan SVM.")

# =========================
# INPUT DATA
# =========================
st.subheader("Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    jenis_kelamin = st.selectbox(
        "Jenis Kelamin",
        ["Laki-laki", "Perempuan"]
    )

    umur = st.number_input(
        "Umur (tahun)",
        min_value=0,
        max_value=120,
        value=20
    )

    jenis_demam = st.selectbox(
        "Jenis Demam",
        ["Demam Ringan", "Demam Sedang", "Demam Tinggi"]
    )

with col2:
    hemoglobin = st.number_input(
        "Hemoglobin (g/dL)",
        min_value=0.0,
        max_value=25.0,
        value=13.0
    )

    hct = st.number_input(
        "Hematokrit (HCT)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    trombosit = st.number_input(
        "Trombosit",
        min_value=0,
        max_value=1000000,
        value=150000
    )

# =========================
# ENCODING
# =========================
# SESUAIKAN DENGAN ENCODING SAAT TRAINING

jk_mapping = {
    "Laki-laki": 1,
    "Perempuan": 0
}

demam_mapping = {
    "Demam Ringan": 0,
    "Demam Sedang": 1,
    "Demam Tinggi": 2
}

input_data = pd.DataFrame({
    "jenis_kelamin": [jk_mapping[jenis_kelamin]],
    "umur": [umur],
    "jenis_demam": [demam_mapping[jenis_demam]],
    "hemoglobin": [hemoglobin],
    "hct": [hct],
    "trombosit": [trombosit]
})

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):

    rf_pred = rf_model.predict(input_data)[0]
    knn_pred = knn_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]

    try:
        rf_prob = rf_model.predict_proba(input_data).max() * 100
    except:
        rf_prob = None

    try:
        knn_prob = knn_model.predict_proba(input_data).max() * 100
    except:
        knn_prob = None

    try:
        svm_prob = svm_model.predict_proba(input_data).max() * 100
    except:
        svm_prob = None

    st.subheader("Hasil Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Random Forest")
        st.write(f"Prediksi : **{rf_pred}**")
        if rf_prob:
            st.write(f"Probabilitas : **{rf_prob:.2f}%**")

    with col2:
        st.info("KNN")
        st.write(f"Prediksi : **{knn_pred}**")
        if knn_prob:
            st.write(f"Probabilitas : **{knn_prob:.2f}%**")

    with col3:
        st.warning("SVM")
        st.write(f"Prediksi : **{svm_pred}**")
        if svm_prob:
            st.write(f"Probabilitas : **{svm_prob:.2f}%**")

    st.subheader("Data Input")

    st.dataframe(input_data)