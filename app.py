import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman (Harus di bagian paling atas!)
st.set_page_config(
    page_title="Prediksi Keterlibatan Siswa", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styling Kustom (CSS Tambahan via Markdown)
st.markdown("""
<style>
    .utama {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        height: 3em;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# 3. Memuat Model dan Scaler (menggunakan caching agar web lebih cepat)
@st.cache_resource
def load_assets():
    model = joblib.load('best_model_decision_tree.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_assets()

# --- HEADER ---
st.title("🎓 Sistem Prediksi Keterlibatan Siswa")
st.markdown("**Platform Prediksi Cerdas** membantu pengajar dan institusi mengidentifikasi tingkat partisipasi *(Engagement Level)* siswa secara proaktif menggunakan Machine Learning.")
st.divider()

# 4. Membagi Antarmuka ke dalam 2 Kolom untuk Estetika
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Profil Demografis")
    with st.container(border=True):
        gender = st.selectbox("👤 Jenis Kelamin", ["M", "F"], format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")
        region = st.selectbox("📍 Wilayah (Region)", encoders['region'].classes_)
        imd_band = st.selectbox("📊 Indeks Deprivasi Majemuk (IMD Band)", encoders['imd_band'].classes_, 
                                help="Ukuran tingkat kesejahteraan lingkungan tempat tinggal siswa.")
        risk_level = st.selectbox("⚠️ Tingkat Risiko (Risk Level)", 
                                  encoders.get('risk_level', type('obj', (object,), {'classes_': ['Safe', 'Low Risk', 'High Risk', 'Very High Risk']})).classes_)

with col2:
    st.subheader("📚 Data Akademik & Aktivitas")
    with st.container(border=True):
        highest_education = st.selectbox("🎓 Pendidikan Tertinggi", encoders['highest_education'].classes_)
        studied_credits = st.number_input("⏳ Total Kredit Belajar", min_value=0, step=10, help="Total SKS/Beban studi yang diambil siswa")
        avg_score = st.slider("📈 Rata-rata Skor Asesmen", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
        total_clicks = st.number_input("🖱️ Total Aktivitas Klik di LMS", min_value=0, step=100, help="Berapa banyak siswa menekan konten edukasi pada portal (LMS)")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Tombol Prediksi Utama (Lebar penuh)
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    # Menggunakan tombol yang dipercantik via CSS
    submit_button = st.button("🚀 LAKUKAN PREDIKSI KETERLIBATAN", use_container_width=True)

st.divider()

# 6. Logika Prediksi
if submit_button:
    with st.spinner('Menganalisis pola historis siswa...'):
        # (a) Format data
        input_data = pd.DataFrame({
            'gender': [gender],
            'region': [region],
            'highest_education': [highest_education],
            'studied_credits': [studied_credits],
            'imd_band': [imd_band],
            'total_clicks': [total_clicks],
            'avg_score': [avg_score],
            'risk_level': [risk_level]
        })
        
        # (b) Lakukan Encoding
        input_data['gender'] = encoders['gender'].transform(input_data['gender'])
        input_data['region'] = encoders['region'].transform(input_data['region'])
        input_data['highest_education'] = encoders['highest_education'].transform(input_data['highest_education'])
        input_data['imd_band'] = encoders['imd_band'].transform(input_data['imd_band'])
        input_data['risk_level'] = encoders['risk_level'].transform(input_data['risk_level'])
        
        # (c) Scaling
        input_scaled = scaler.transform(input_data)
        
        # (d) Prediksi
        hasil_prediksi = model.predict(input_scaled)
        status_hasil = encoders['engagement_level'].inverse_transform(hasil_prediksi)[0]
        
    # (e) Menampilkan Hasil dengan UI yang Menarik
    st.subheader("💡 Hasil Analisis")
    
    if status_hasil == 'High':
        st.balloons()
        st.success("Tingkat Keterlibatan Diprediksi: **TINGGI (High Engagement)** 🌟")
        st.info("Siswa ini diproyeksikan sangat proaktif dalam pembelajaran. Pertahankan terus!")
    elif status_hasil == 'Medium':
        st.info("Tingkat Keterlibatan Diprediksi: **MENENGAH (Medium Engagement)** 👍")
        st.write("Siswa ini memiliki partisipasi yang cukup standar. Bisa diberikan motivasi tambahan sesekali.")
    else:
        st.warning("Tingkat Keterlibatan Diprediksi: **RENDAH (Low Engagement)** ⚠️")
        st.error("**Tindakan Dibutuhkan:** Siswa ini memiliki indikasi partisipasi yang sangat rendah dan berrisiko tertinggal. Segera hubungi siswa untuk bimbingan!")
        
    # Menampilkan Metric Card Rangkuman Profil
    st.markdown("<br><p class='big-font'>Ringkasan Eksekutif:</p>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(label="Rerata Skor", value=f"{avg_score:.1f}/100", delta="Akademik")
    m_col2.metric(label="Aktivitas Platform", value=f"{int(total_clicks)} Klik", delta="Keaktifan")
    m_col3.metric(label="Tingkat Risiko Info", value=risk_level, delta="Profil Utama", delta_color="off")
