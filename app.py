import numpy as np
from PIL import Image,  ImageOps
import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Skenario5.h5')
    return model


model = load_model()

st.title("Klasifikasi Gerakan Yoga")

file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):
    size = (170, 170)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Tolong upload gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    labels = ['goddess', 'plank', 'tree', 'warrior2', 'downdog']
    string = "Prediksi Gambar : "+labels[np.argmax(predictions)]
    if predictions[0][0] == 1:
        txt = st.text_area('Manfaat Gerakan', '''
            Gerakan goddess yang biasa disebut dengan half squat ini memiliki kegunaan yang sangat mempengaruhi bagi tubuh yaitu memperkuat bagian kaki, otot pada bokong, dan otot bagian perut
            ''')
    elif predictions[0][1] != 0:
        txt = st.text_area('Manfaat Gerakan', '''
            Gerakan plank memiliki kegunaan untuk memperkuat otot inti dari tubuh, membantu pembentukan otot perut, dan dapat memulihkan postur tubuh
            ''')
    elif predictions[0][2] != 0:
        txt = st.text_area('Manfaat Gerakan', '''
            Gerakan tree ini memiliki kegunaan untuk melatih keseimbangan dan gerakan ini sangat berguna untuk ibu hamil terutama pada janin yang ada di dalam perut mendapatkan asupan oksigen yang sangat baik
            ''')
    elif predictions[0][3] != 0:
        txt = st.text_area('Manfaat Gerakan', '''
            Gerakan warior 2 atau virabhadasarana II memiliki kegunaan untuk memperkuat otot kaki, perut, lengan dan dapat memperkuat daya tahan serta meningkatkan energi yang ada dalam tubuh
            ''')
        txt = st.text('')
    else:
        txt = st.text_area('Manfaat Gerakan', '''
            Gerakan downdog atau downward dog ini memiliki kegunaan yang dapat membantu tulang lengan yang biasanya rentan terkena penyakit tulang yaitu osteophorosist
            ''')
    st.success(string)
