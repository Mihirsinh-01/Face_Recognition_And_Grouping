from distutils.command.upload import upload
from turtle import width
import face_recognition as fr
import cv2
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import numpy as np


def show_images(imgs):
    st.image(imgs, width=100)


# Title
st.title("MyPick")
st.subheader("Project on Face Recognition and Grouping")


st.sidebar.title("Project Info")


image = Image.open('static/logo.jpg')
st.image(image, caption='Insitute of Technology - Nirma University')


# Sidebar Components
st.sidebar.write("**Student Name:** Vaja Mihirsinih Vikramsinh")
st.sidebar.write("**Roll No.:** 18BCE251")
st.sidebar.write("**Project Guide:** Prof. Kruti Lavingia")
st.sidebar.write("**Semester:** 8")

st.sidebar.info(
    "A Project based on Face Detection and Grouping to separate the individuals images")

st.sidebar.write("**Tolerance Value**")
tolerance = st.sidebar.slider(
    "Less Value - More strict comparison", min_value=0, max_value=100, value=60)

uploaded_files = st.file_uploader("Upload all the images",
                                  accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:

    # Displaying all uploaded Files
    st.subheader("Uploaded Images")
    st.image(uploaded_files, width=150)

    print(uploaded_files[0])

    FACES = {
        'face_location': [],
        'filepath': [],
        'face_image': [],
        'encoding': []
    }

    for file in uploaded_files:

        # Loading the image
        image = fr.load_image_file(file)
        face_locations = fr.face_locations(image)

        for location in face_locations:
            (top, right, bottom, left) = location
            face_image = image[top:bottom, left:right]

            encoding = fr.face_encodings(face_image)
            if encoding != []:
                FACES['face_location'].append(location)
                FACES['filepath'].append(file)
                FACES['face_image'].append(face_image)
                FACES['encoding'].append(encoding[0])

    st.subheader("All the Faces Detected")
    # st.write("Total {} faces Found".format(len(FACES['face_image'])))
    show_images(FACES['face_image'])

    face_encoding = FACES['encoding']

    r = list(range(len(face_encoding)))
    face_cluster = []
    while len(r) > 0:
        face_cluster.append([])
        result = fr.compare_faces(
            face_encoding, face_encoding[r[0]], tolerance=tolerance/100)
        for index, boo in enumerate(result):
            if boo and index in r:
                face_cluster[-1].append(index)
                r.remove(index)

    print("Face Clusters ", face_cluster)

    for index, cluster in enumerate(face_cluster):
        faces = [FACES['face_image'][index] for index in cluster]
        st.write("Person-", index+1)
        show_images(faces)
        components.html("<hr><hr>")
