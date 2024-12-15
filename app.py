# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Scene Classification App",
    page_icon="🌄",
    layout="wide"
)

# Title and description
st.title("Scene Classification using VGG16")
st.write("Upload an image of a scene to classify it into one of six categories: buildings, forest, glacier, mountain, sea, or street.")

def create_vgg16_model(input_shape=(224, 224, 3), num_classes=6):
    """Recreate the model architecture"""
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

@st.cache_resource
def load_classification_model():
    """Load the trained VGG16 model"""
    try:
        # Create model architecture
        model = create_vgg16_model()
        
        # Load saved weights
        weights_path = 'vgg16.weights.h5'
        if not os.path.exists(weights_path):
            st.error(f"Weights file not found at {weights_path}")
            return None
        
        model.load_weights(weights_path)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image to 224x224
        image = image.resize((224, 224))
        # Convert to array
        img_array = img_to_array(image)
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess for VGG16
        preprocessed_img = preprocess_input(img_array)
        return preprocessed_img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Make prediction on preprocessed image"""
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    try:
        # Preprocess image
        preprocessed_img = preprocess_image(image)
        if preprocessed_img is None:
            return None, None, None
        
        # Make prediction
        predictions = model.predict(preprocessed_img, verbose=0)
        
        # Get top prediction
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        # Get all class probabilities
        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(class_names, predictions[0])
        }
        
        return predicted_class, confidence, class_probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Add some CSS styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the weights file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a scene to classify"
    )
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Scene", use_column_width=True)
            
            # Make prediction
            predicted_class, confidence, class_probabilities = predict_scene(model, image)
            
            if predicted_class is not None:
                # Display results
                with col2:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("Prediction Results")
                    st.write(f"**Predicted Scene:** {predicted_class.title()}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display probability bar chart
                    st.subheader("Class Probabilities")
                    probabilities_df = pd.DataFrame({
                        'Class': list(class_probabilities.keys()),
                        'Probability': list(class_probabilities.values())
                    })
                    probabilities_df['Class'] = probabilities_df['Class'].str.title()
                    
                    # Sort probabilities in descending order
                    probabilities_df = probabilities_df.sort_values('Probability', ascending=True)
                    
                    # Create a more visually appealing chart
                    chart = st.bar_chart(
                        probabilities_df.set_index('Class'),
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
