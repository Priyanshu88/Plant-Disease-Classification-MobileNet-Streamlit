import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the function to preprocess the image


def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

# Define the function to make a prediction


def make_prediction(model, image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0]

    # Replace with your class names
    class_names = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___hea']

    # Get the top 3 class indices with highest probabilities
    top_3_indices = np.argsort(prediction)[::-1][:3]

    top_3_classes = [class_names[idx] for idx in top_3_indices]
    top_3_probabilities = prediction[top_3_indices]

    return top_3_classes, top_3_probabilities


def main():
    # Load the model
    model_path = "plantdiseasemobilenet12epoch.h5"
    # Customize the optimizer as per your requirements
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Set the app title
    st.title("Plant Disease Classification")

    # Upload and classify the image
    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image",
                 use_column_width=True)

        # Preprocess and make prediction
        image = Image.open(uploaded_image)
        top_3_classes, top_3_probabilities = make_prediction(model, image)

        # Display the top 3 prediction results
        st.subheader("Top 3 Predictions:")
        for i in range(3):
            st.write(f"{top_3_classes[i]}: {top_3_probabilities[i]:.4f}")


if __name__ == "__main__":
    main()
