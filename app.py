# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Scene Classification App",
    page_icon="🌄",
    layout="wide"
)

# Title and description
st.title("Scene Classification using VGG16")
st.write("Upload an image of a scene to classify it into one of six categories: buildings, forest, glacier, mountain, sea, or street.")

@st.cache_resource
def load_classification_model():
    """Load the trained VGG16 model"""
    try:
        model = load_model('best_vgg16.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert to array
    img_array = img_to_array(image)
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for VGG16
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

def predict_scene(model, image):
    """Make prediction on preprocessed image"""
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Preprocess image
    preprocessed_img = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(preprocessed_img)
    
    # Get top prediction
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    # Get all class probabilities
    class_probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, predictions[0])
    }
    
    return predicted_class, confidence, class_probabilities

def main():
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display uploaded image
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Scene", use_column_width=True)
        
        # Make prediction
        predicted_class, confidence, class_probabilities = predict_scene(model, image)
        
        # Display results
        with col2:
            st.subheader("Prediction Results")
            st.write(f"**Predicted Scene:** {predicted_class.title()}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Display probability bar chart
            st.subheader("Class Probabilities")
            probabilities_df = pd.DataFrame({
                'Class': list(class_probabilities.keys()),
                'Probability': list(class_probabilities.values())
            })
            probabilities_df['Class'] = probabilities_df['Class'].str.title()
            st.bar_chart(probabilities_df.set_index('Class'))

if __name__ == "__main__":
    main()
