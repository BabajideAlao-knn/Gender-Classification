from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from PIL import Image
# import pickle
import joblib

gender_model = joblib.load('gender_model.pkl')

st.set_page_config(
    layout='wide',
    page_title='Gender Prediction',
    # page_icon='img.jfif',
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            footer:after {
                          content:'Created by Babajide Alao'; 
                          visibility: visible;
                          display: block;
                          position: relative;
                          #background-color: white;
                          padding: 4px;
                          top: 2px;
                          }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def set_background(png_file):
    bin_str = png_file
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


long_hair = st.selectbox('Do you have a Long hair?', ["Yes", "No"])
nose_wide = st.selectbox('Do you have a wide nose?', ["Yes", "No"])
nose_long = st.selectbox('Do you have a Long nose?', ["Yes", "No"])
lips_thin = st.selectbox('Do you have thin lips?', ["Yes", "No"])
distance_nose_to_lip_long = st.selectbox('Is the distance from the nose to the lip long?', ["Yes", "No"])
forehead_width_cm = st.number_input('Forehead width (cm):', min_value=14, max_value=16, value=14)
forehead_height_cm = st.number_input('Forehead height (cm):', min_value=5, max_value=8, value=5)


def predict(long_hair, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long, forehead_width_cm,
            forehead_height_cm):
    if long_hair == "Yes":
        long_hair = 1
    else:
        long_hair = 0

    if nose_wide == "Yes":
        nose_wide = 1
    else:
        nose_wide = 0

    if nose_long == "Yes":
        nose_long = 1
    else:
        nose_long = 0

    if lips_thin == "Yes":
        lips_thin = 1
    else:
        lips_thin = 0

    stdscale = MinMaxScaler()
    x_data = stdscale.fit_transform(pd.DataFrame
                                    [
                                        long_hair, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long, forehead_width_cm, forehead_height_cm],
                                    columns=['long_hair', 'nose_wide', 'nose_long', 'lips_thin',
                                             'distance_nose_to_lip_long', 'forehead_width_cm', 'forehead_height_cm'])

    prediction = gender_model.predict(x_data)

    return prediction


if st.button('Predict Gender'):
    Gender = predict(long_hair, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long, forehead_width_cm,
                     forehead_height_cm)
    if Gender == 0:
        Gender = "Female"
    else:
        Gender = "Male"
    st.write('The predicted Gender is: {}'.format(Gender))

st.title("""
        This app was developed by:
        """)
st.text('- Babajide Alao (https://github.com/BabajideAlao-knn)')
