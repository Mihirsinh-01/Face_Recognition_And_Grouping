import face_recognition as fr
import cv2
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import numpy as np

# Title
st.title("MyPick")
st.subheader("Project on Face Recognition and Grouping")


st.sidebar.title("Project Info")


image = Image.open('static/logo.jpg')
st.image(image, caption='Insitute of Technology - Nirma University')

st.sidebar.write("**Student Name:** Vaja Mihirsinih Vikramsinh")
st.sidebar.write("**Roll No.:** 18BCE251")
st.sidebar.write("**Project Guide:** Prof. Kruti Lavingia")
st.sidebar.write("**Semester:** 8")

st.sidebar.info(
    "A Project based on Face Detection and Grouping to separate the individuals images")

uploaded_files = st.file_uploader("Upload all the images",
                                  accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    all_face_enc = []
    new_faces = []
    duplicate_faces = []
    for img in uploaded_files:

        img1 = Image.open(img)
        rgb_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)
        all = fr.face_encodings(rgb_img1)
        locations = fr.face_locations(rgb_img1)

        for (top, right, bottom, left), face in zip(locations, all):

            cv2.rectangle(rgb_img1, (left, top),
                          (right, bottom), (0, 0, 255), 2)

            cropped_img = rgb_img1[top:bottom, left:right]
            # st.write("The length of Array is", len(new_faces))
            if len(all_face_enc) == 0:
                bgr_img1 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                # st.image(bgr_img1, width=500)
                new_faces.append(bgr_img1)
            else:
                result = fr.compare_faces(all_face_enc, face)
                if not any(result):
                    bgr_img1 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                    new_faces.append(bgr_img1)
                else:
                    bgr_img1 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                    duplicate_faces.append(bgr_img1)

            all_face_enc.append(face)

    cnt = len(new_faces)
    st.write('Total {} faces found in the image'.format(cnt))
    st.header("Unique Faces")
    for unique in new_faces:
        st.image(unique, width=100)
    components.html("<hr>")

    st.header("Duplicate Faces")
    for dupl in duplicate_faces:
        st.image(dupl, width=100)
    components.html("<hr>")


# img1 = cv2.imread("images/16.jpeg")
# rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# # cv2.imshow("img1", rgb_img1)
# img1_enc = fr.face_encodings(rgb_img1)[0]

# img2 = cv2.imread("images/4.jpeg")
# rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# # cv2.imshow("img2", rgb_img2)
# img2_enc = fr.face_encodings(rgb_img2)[1]

# print("Length is", len(fr.face_encodings(rgb_img2)))

# # result = fr.compare_faces([img1_enc], img2_enc)
# # print(result)

# # cv2.waitKey(0)
