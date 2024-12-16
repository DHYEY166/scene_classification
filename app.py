import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from PIL import Image
import os

def create_model():
    """Create identical model architecture"""
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization(name='bn_1')(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='dense_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='dense_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    x = BatchNormalization(name='bn_3')(x)
    
    outputs = Dense(6, activation='softmax', kernel_regularizer=l2(0.01), name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        # Create model architecture
        model = create_model()
        model.summary()  # Print summary to verify layer count
        
        # Load weights
        weights_path = 'scene_classifier.weights.h5'
        if not os.path.exists(weights_path):
            st.error(f"Weights not found at: {weights_path}")
            return None
        
        # Load weights without by_name parameter
        model.load_weights(weights_path)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match training dimensions
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)
        
        return processed_img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Make prediction on image"""
    try:
        # Class names must match training order
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Preprocess image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Create results dictionary
        results = {
            class_name: float(prob)
            for class_name, prob in zip(classes, predictions[0])
        }
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Add sidebar with info
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This app uses a VGG16-based model trained on scene images.
        The model classifies images into six categories.
        """)
        
        # Model details
        if st.checkbox("Show Technical Details"):
            st.write("""
            - Base Model: VGG16
            - Input Size: 224x224
            - Classes: 6
            - Training: Transfer Learning
            """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a scene to classify"
    )
    
    # Process uploaded image
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Display uploaded image
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Make and display prediction
        with col2:
            st.subheader("Prediction Results")
            
            with st.spinner('Analyzing image...'):
                predictions = predict_scene(model, image)
            
            if predictions:
                # Get top prediction
                top_class = max(predictions.items(), key=lambda x: x[1])
                
                # Display results in a card-like container
                st.markdown("""
                <style>
                .prediction-box {
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #f0f2f6;
                    margin-bottom: 20px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Predicted Scene: **{top_class[0].title()}**")
                st.markdown(f"### Confidence: **{top_class[1]:.2%}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show all class probabilities
                st.subheader("Class Probabilities")
                
                # Create DataFrame for visualization
                df = pd.DataFrame(
                    list(predictions.items()),
                    columns=['Class', 'Probability']
                )
                
                # Sort by probability
                df = df.sort_values('Probability', ascending=True)
                
                # Display bar chart
                st.bar_chart(df.set_index('Class'))

if __name__ == "__main__":
    main()
