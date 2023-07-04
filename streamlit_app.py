import streamlit as st
import face_recognition
from PIL import Image

def calculate_similarity(image1, image2):
    # 画像読み込み
    img1 = face_recognition.load_image_file(image1)
    img2 = face_recognition.load_image_file(image2)

    # 顔の特徴を抽出
    encoding1 = face_recognition.face_encodings(img1)[0]
    encoding2 = face_recognition.face_encodings(img2)[0]

    # 類似度計算
    similarity = face_recognition.face_distance([encoding1], encoding2)[0]

    return similarity

# Streamlitアプリケーションの設定
st.title("顔類似度計算アプリ")
st.write("2枚の画像の類似度を計算します。")

# 画像アップロード
uploaded_file1 = st.file_uploader("画像1を選択してください", type=['jpg', 'jpeg', 'png'])
uploaded_file2 = st.file_uploader("画像2を選択してください", type=['jpg', 'jpeg', 'png'])

if uploaded_file1 is not None and uploaded_file2 is not None:
    # 画像表示
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    st.image([image1, image2], caption=['Image 1', 'Image 2'], width=300)

    # 類似度計算
    similarity = calculate_similarity(uploaded_file1, uploaded_file2)
    st.write(f"類似度: {similarity:.2f}")
