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
        ["DBD", "DD", "DSS"]
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
    "Laki-laki": 0,
    "Perempuan": 1
}

demam_mapping = {
    "DBD": 0,
    "DD": 1,
    "DSS": 2
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

    predictions = [rf_pred, knn_pred, svm_pred]

    majority_vote = max(set(predictions), key=predictions.count)

    label_mapping = {
        0: "Pendek",
        1: "Panjang"
    }

    rf_label = label_mapping[rf_pred]
    knn_label = label_mapping[knn_pred]
    svm_label = label_mapping[svm_pred]

    majority_vote_label = label_mapping[majority_vote]

    vote_prob = predictions.count(majority_vote) / len(predictions) * 100

    st.subheader("Hasil Majority Voting")
    st.success(f"Hasil Akhir: **{majority_vote_label}**")
    st.info(f"Probabilitas Voting: **{vote_prob:.2f}%**")


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
        st.write(f"Prediksi : **{rf_label}**")
        if rf_prob:
            st.write(f"Probabilitas : **{rf_prob:.2f}%**")

    with col2:
        st.info("KNN")
        st.write(f"Prediksi : **{knn_label}**")
        if knn_prob:
            st.write(f"Probabilitas : **{knn_prob:.2f}%**")

    with col3:
        st.warning("SVM")
        st.write(f"Prediksi : **{svm_label}**")
        if svm_prob:
            st.write(f"Probabilitas : **{svm_prob:.2f}%**")

    st.subheader("Data Input")

    display_data = pd.DataFrame({
        "Jenis Kelamin": [jenis_kelamin],
        "Umur": [umur],
        "Jenis Demam": [jenis_demam],
        "Hemoglobin": [hemoglobin],
        "HCT": [hct],
        "Trombosit": [trombosit]
    })

    st.dataframe(display_data)