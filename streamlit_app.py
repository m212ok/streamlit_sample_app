import streamlit as st
from PIL import Image, ImageDraw
import face_recognition

def draw_rectangle_with_landmarks(image, face_locations, face_landmarks):
    draw = ImageDraw.Draw(image)

    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
        # 顔の矩形を描画
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)

        # 顔のランドマークを描画
        for landmark in landmarks.values():
            draw.line(landmark, fill="red", width = 5)

def draw_rectangle(image, top, right, bottom, left):
    draw = ImageDraw.Draw(image)
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)

def calculate_similarity(image1, image2):
    # 画像を読み込み、顔の位置と特徴を抽出
    img1 = face_recognition.load_image_file(image1)
    img2 = face_recognition.load_image_file(image2)
    face_locations1 = face_recognition.face_locations(img1)
    face_locations2 = face_recognition.face_locations(img2)

    if len(face_locations1) == 0 or len(face_locations2) == 0:
        # 顔が検出されない場合の処理
        return 0, [], []

    # 顔部分の特徴を抽出
    encodings1 = face_recognition.face_encodings(img1, face_locations1)
    encodings2 = face_recognition.face_encodings(img2, face_locations2)
    
    # 顔のランドマークを抽出
    landmarks1 = face_recognition.face_landmarks(img1, face_locations1)
    landmarks2 = face_recognition.face_landmarks(img2, face_locations2)
    
    # 類似度を計算
    face_distance = face_recognition.face_distance(encodings1, encodings2[0])
    similarity = 1 - face_distance[0]
    
    return similarity, face_locations1, face_locations2, encodings1, encodings2, landmarks1, landmarks2

# Streamlitアプリケーションの設定
st.title("顔の類似度計算アプリ")
st.write("2枚の画像に写る人物の顔を検出し、類似度を計算します。")

# 画像アップロード
uploaded_file1 = st.file_uploader("画像1を選択してください", type=['jpg', 'jpeg', 'png'])
uploaded_file2 = st.file_uploader("画像2を選択してください", type=['jpg', 'jpeg', 'png'])

if uploaded_file1 is not None and uploaded_file2 is not None:
    # 画像表示
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    st.image([image1, image2], caption=['Image 1', 'Image 2'], width=300)

    # 類似度計算
    similarity, face_locations1, face_locations2, face_encodings1, face_encodings2, face_landmarks1, face_landmarks2 = calculate_similarity(uploaded_file1, uploaded_file2)
    st.write(f"類似度: {similarity:.2f}")

    # 顔の矩形を表示
    image1_with_rectangles = image1.copy()
    image2_with_rectangles = image2.copy()

    draw_rectangle_with_landmarks(image1_with_rectangles, face_locations1, face_landmarks1)
    draw_rectangle_with_landmarks(image2_with_rectangles, face_locations2, face_landmarks2)
    
    st.image(image1_with_rectangles, caption='Image 1 with Face Rectangles & Face Landmark', width=300)
    st.image(image2_with_rectangles, caption='Image 2 with Face Rectangles & Face Landmark', width=300)