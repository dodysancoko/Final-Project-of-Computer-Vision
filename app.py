import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO

# =======================
# KONFIGURASI HALAMAN
# =======================
st.set_page_config(page_title="Komparasi YOLOv8 & YOLOv9 - Deteksi Daun Tomat", layout="wide")
st.title("🍅 Komparasi Deteksi Penyakit Daun Tomat: YOLOv8s vs YOLOv9s")

# =======================
# DESKRIPSI DAN LATAR BELAKANG
# =======================
st.markdown("""
Aplikasi ini dibuat untuk **membandingkan performa dua model deep learning** dalam mendeteksi **penyakit pada daun tomat**, yaitu:
- **YOLOv8s** – Efisien dan cepat
- **YOLOv9s** – Stabil dan akurat

### 🧪 Latar Belakang:
Tomat merupakan salah satu komoditas penting di Indonesia. Namun, tanaman tomat sangat rentan terhadap penyakit daun yang bisa menyebabkan kerugian besar.  
Deteksi dini sangat penting, dan teknologi deep learning seperti **YOLO (You Only Look Once)** sangat efektif untuk tugas ini.

Penelitian kami membandingkan **YOLOv8s** dan **YOLOv9s** menggunakan **dataset penyakit daun tomat dari lingkungan alami**.  
Model diuji pada **10 kelas penyakit**, di antaranya:
- Tomato bacterial spot  
- Tomato early blight  
- Tomato yellow leaf curl virus  
- Tomato mosaic virus  
*(dan lainnya)*

Model YOLOv9s unggul dalam **akurasi dan stabilitas pelatihan**, sementara YOLOv8s unggul dalam **efisiensi waktu pelatihan**.

### 👨‍💻 Tim Pengembang:
- **Gede Bagus Krishnanditya M. (1301223088)**  
- **Raka Aditya Waluya (1301220192)**  
- **Dody Adi Sancoko (1301223071)**
""")

# =======================
# PENJELASAN KELAS PENYAKIT
# =======================
with st.expander("🩺 Daftar Penyakit Daun Tomat yang Dapat Deteksi"):
    st.markdown("""
    Model mendeteksi **10 kelas** kondisi daun tomat, yaitu:

    1. **Tomato Bacterial Spot** – Bercak gelap kecil menyebar pada daun.
    2. **Tomato Early Blight** – Bercak coklat besar dengan pola cincin.
    3. **Tomato Late Blight** – Daun basah kehijauan berubah menjadi hitam.
    4. **Tomato Leaf Mold** – Lapisan kuning/hijau pucat di atas daun.
    5. **Tomato Septoria Leaf Spot** – Bercak abu-abu kecil dengan tepi gelap.
    6. **Tomato Spider Mites (Two-Spotted)** – Bercak putih kecil + jaring halus.
    7. **Tomato Target Spot** – Bercak coklat bundar dengan pusat terang.
    8. **Tomato Yellow Leaf Curl Virus** – Daun keriting dan kuning pucat.
    9. **Tomato Mosaic Virus** – Daun belang hijau-kuning dengan bentuk tak normal.
    10. **Tomato Healthy** – Daun sehat tanpa gejala penyakit.
    """)

# =======================
# LOAD MODELS
# =======================
@st.cache_resource
def load_models():
    model_yolov8 = YOLO("yolov8s.pt")
    model_yolov9 = YOLO("yolov9s.pt")
    return model_yolov8, model_yolov9

model_yolov8, model_yolov9 = load_models()

# =======================
# MAPPING ID KE LABEL
# =======================
id_to_label = {
    0: "Tomato Bacterial Spot",
    1: "Tomato Early Blight",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Leaf Spot",
    5: "Tomato Spider Mites Two-Spotted",
    6: "Tomato Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Tomato Healthy"
}

# =======================
# MAPPING PENJELASAN KELAS
# =======================
penjelasan_kelas = {
    "Tomato Bacterial Spot": "Bercak gelap kecil menyebar pada daun.",
    "Tomato Early Blight": "Bercak coklat besar dengan pola cincin.",
    "Tomato Late Blight": "Daun basah kehijauan berubah menjadi hitam.",
    "Tomato Leaf Mold": "Lapisan kuning/hijau pucat di atas daun.",
    "Tomato Septoria Leaf Spot": "Bercak abu-abu kecil dengan tepi gelap.",
    "Tomato Spider Mites Two-Spotted": "Bercak putih kecil + jaring halus.",
    "Tomato Target Spot": "Bercak coklat bundar dengan pusat terang.",
    "Tomato Yellow Leaf Curl Virus": "Daun keriting dan kuning pucat.",
    "Tomato Mosaic Virus": "Daun belang hijau-kuning dengan bentuk tak normal.",
    "Tomato Healthy": "Daun sehat tanpa gejala penyakit."
}

# =======================
# UPLOAD GAMBAR
# =======================
st.markdown("## 📤 Upload Gambar Daun Tomat")
uploaded_file = st.file_uploader("Upload gambar daun tomat (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang Diupload", width=400)

    # Simpan gambar sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        image.save(temp_path)

    # =======================
    # INFERENSI / PREDIKSI
    # =======================
    with st.spinner("🔍 YOLOv8s & YOLOv9s sedang mendeteksi..."):
        result_v8 = model_yolov8.predict(temp_path, conf=0.25)[0]
        result_v9 = model_yolov9.predict(temp_path, conf=0.25)[0]

    # =======================
    # TAMPILKAN HASIL DUA MODEL
    # =======================
    st.markdown("## 🔬 Hasil Komparasi Deteksi")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("YOLOv8s 🔵 (Cepat & Efisien)")
        st.image(result_v8.plot(), caption="Hasil Deteksi - YOLOv8s", width=500)
        st.markdown(f"### 🧾 Objek Terdeteksi: {len(result_v8.boxes)} objek")
        for box in result_v8.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = id_to_label.get(cls_id, f"Unknown Class ({cls_id})")
            deskripsi = penjelasan_kelas.get(label, "Deskripsi tidak ditemukan.")
            st.markdown(f"- **{label}** ({conf:.2f})  \n  _{deskripsi}_")

    with col2:
        st.subheader("YOLOv9s 🟢 (Akurat & Stabil)")
        st.image(result_v9.plot(), caption="Hasil Deteksi - YOLOv9s", width=500)
        st.markdown(f"### 🧾 Objek Terdeteksi: {len(result_v9.boxes)} objek")
        for box in result_v9.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = id_to_label.get(cls_id, f"Unknown Class ({cls_id})")
            deskripsi = penjelasan_kelas.get(label, "Deskripsi tidak ditemukan.")
            st.markdown(f"- **{label}** ({conf:.2f})  \n  _{deskripsi}_")
