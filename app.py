# 12/20 11:39
import streamlit as st
import tensorflow as tf
import numpy as np
import keras
from PIL import Image
from keras.models import load_model
import keras.utils as image



def streamlit():
    
    model = keras.models.load_model('mnist.hdf5')
    
    labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

    
    try:
        uploaded_file = st.sidebar.file_uploader(label = 'Upload an image')
        class_btn = st.sidebar.button("Classify")
        
        
        
        if uploaded_file is not None: 

            image = Image.open(uploaded_file)
            image = image.resize((224, 224))
            st.image(image, caption='Uploaded Image')
            
            image = tf.keras.utils.load_img(uploaded_file, color_mode= "grayscale", target_size=(28, 28))
            img_np = tf.keras.utils.img_to_array(image)
            img_np = np.expand_dims(img_np, axis=0)
            

            img_np = np.array(image) / 255 # noramlize
            img_np.reshape(1, 28, 28, 1)
            img_np = np.array([img_np]) # add dimesion
            
            if class_btn == True:
                prediction = model.predict(img_np)
                pred_list = list(prediction)
                pred_list = [round(pred_list[0][i]) for i in range(10) ]
                pred = pred_list.index(1)
                st.write("Prediction of image is :")
                st.write(labels[pred])
                st.success('Classified')
        if  uploaded_file is  None: 
                st.title('handwritten digits Classifier')

                st.subheader("Welcome to this simple web application that classifies handwritten digits from 0 to 9")
        else:
            pass        
    except:
        
        st.warning("format not supported or  you are trying to upload an image over an existing image.", icon="🚨")
        
        st.info("Rerun the app please!")
   
    
streamlit()