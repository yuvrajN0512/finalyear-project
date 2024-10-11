import streamlit as st
import tensorflow as tf
import numpy as np

# Solutions for diseases in English, Tamil, and Hindi
solutions = {
    'Apple___Apple_scab': {
        'English': "Use Captan (2 grams per liter of water). Apply 500 liters per hectare.",
        'Tamil': "காப்டன் (நீர் ஒரு லிட்டருக்கு 2 கிராம்) பயன்படுத்தவும். ஒரு ஹெக்டேருக்கு 500 லிட்டர்கள் தெளிக்கவும்.",
        'Hindi': "कैप्टान (2 ग्राम प्रति लीटर पानी) का उपयोग करें। 500 लीटर प्रति हेक्टेयर छिड़काव करें।"
    },
    # Add solutions for the other 37 diseases in the same format
    # Example for the next disease:
    'Apple___Black_rot': {
        'English': "Use Thiophanate-methyl (2 grams per liter of water). Apply 600 liters per hectare.",
        'Tamil': "தியோபானேட்-மெதிலால் (நீர் ஒரு லிட்டருக்கு 2 கிராம்) பயன்படுத்தவும். 600 லிட்டர்கள் தெளிக்கவும்.",
        'Hindi': "थियोफेनेट-मिथाइल (2 ग्राम प्रति लीटर पानी) का उपयोग करें। 600 लीटर प्रति हेक्टेयर छिड़काव करें।"
    },
    # Continue adding solutions for all 38 classes
}

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset consists of 87K images of healthy and diseased crop leaves categorized into 38 different classes.
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image") and test_image:
        st.image(test_image, width=4, use_column_width=True)

    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        predicted_disease = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_disease}")

        # Language Selection Buttons
        st.write("Choose a language for the solution:")
        if predicted_disease in solutions:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("English"):
                    st.write(f"Solution in English: {solutions[predicted_disease]['English']}")
            with col2:
                if st.button("Tamil"):
                    st.write(f"Solution in Tamil: {solutions[predicted_disease]['Tamil']}")
            with col3:
                if st.button("Hindi"):
                    st.write(f"Solution in Hindi: {solutions[predicted_disease]['Hindi']}")
        else:
            st.error("No solution available for this disease.")
