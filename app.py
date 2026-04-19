import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Sayfa Ayarları
st.set_page_config(page_title="CariesVision AI", page_icon="🦷")
st.title("🦷 CariesVision AI: Dental Diagnostic Support")
st.markdown("---")

# 2. Modeli Yükle (Pro modelini kullanıyoruz)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('dis_modeli_pro.h5')

model = load_my_model()

# 3. Fotoğraf Yükleme Alanı
st.sidebar.header("Navigation")
uploaded_file = st.file_uploader("Bir diş fotoğrafı yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Görüntüyü ekranda göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Fotoğraf', use_column_width=True)
    
    # AI için hazırlık
    st.write("🔄 Yapay zeka analiz ediyor...")
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    prediction = model.predict(img_array)
    
    if prediction[0][0] < 0.5:
        confidence = (1 - prediction[0][0]) * 100
        st.error(f"⚠️ TEŞHİS: ÇÜRÜK TESPİT EDİLDİ! (Güven: %{confidence:.2f})")
    else:
        confidence = prediction[0][0] * 100
        st.success(f"✅ TEŞHİS: SAĞLIKLI DİŞ (Güven: %{confidence:.2f})")
else:
    st.info("Lütfen sol taraftan veya yukarıdan bir fotoğraf yükleyin.")