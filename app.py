# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import numpy as np
import pandas as pd
from PIL import Image
import os

# Basic Streamlit config
st.set_page_config(
    page_title="Scene Classification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title and description
st.title("Scene Classification using VGG16")
st.write("Upload an image to classify scenes into: buildings, forest, glacier, mountain, sea, or street")

# Debug info in sidebar
with st.sidebar:
    st.subheader("Debug Information")
    st.write(f"TensorFlow version: {tf.__version__}")

def create_model(weights='imagenet'):
    """Create the VGG16 model architecture with proper weight initialization"""
    try:
        # Load VGG16 with imagenet weights
        base_model = VGG16(
            weights=weights,
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create top model
        inputs = base_model.input
        x = base_model.output
        
        # Add custom layers
        x = GlobalAveragePooling2D(name='custom_gap')(x)
        x = BatchNormalization(name='custom_bn_1')(x)
        
        x = Dense(512, activation='relu', name='custom_dense_1')(x)
        x = Dropout(0.3, name='custom_dropout_1')(x)
        x = BatchNormalization(name='custom_bn_2')(x)
        
        x = Dense(256, activation='relu', name='custom_dense_2')(x)
        x = Dropout(0.2, name='custom_dropout_2')(x)
        x = BatchNormalization(name='custom_bn_3')(x)
        
        outputs = Dense(6, activation='softmax', name='predictions')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

def try_load_custom_weights(model):
    """Try to load custom weights or return initialized model"""
    weight_files = [
        'vgg16_complete.weights.h5',
        'best_vgg16.keras',
        'best_vgg16.h5'
    ]
    
    # Check if any weight files exist
    available_files = [f for f in weight_files if os.path.exists(f)]
    
    if not available_files:
        st.info("No weight files found. Using base model with ImageNet weights.")
        return False
        
    # Try loading available weights
    for weights_path in available_files:
        try:
            st.info(f"Attempting to load weights from: {weights_path}")
            model.load_weights(
                weights_path,
                by_name=True,
                skip_mismatch=True
            )
            return True
        except Exception as e:
            st.warning(f"Could not load weights from {weights_path}")
            continue
            
    st.warning("Could not load any weights. Using base model.")
    return False

@st.cache_resource
def load_model():
    """Load model with proper weight initialization"""
    try:
        # Create model with ImageNet weights for base layers
        st.info("Creating model with ImageNet weights...")
        model = create_model(weights='imagenet')
        
        if model is None:
            return None
        
        # Try to load custom top layer weights
        weights_loaded = try_load_custom_weights(model)
        
        if not weights_loaded:
            st.warning("Could not load custom weights. Using initialization weights.")
            
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
            
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array
        img_array = np.array(image)
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for VGG16
        preprocessed = preprocess_input(img_array)
        
        return preprocessed
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, image):
    """Make prediction on image"""
    try:
        # Class names
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None
            
        # Make prediction
        predictions = model.predict(preprocessed, verbose=0)
        
        # Create results dictionary
        results = {
            class_name: float(pred)
            for class_name, pred in zip(classes, predictions[0])
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    """Main app function"""
    try:
        # Load model
        model = load_model()
        if model is None:
            return
            
        # Create file uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Create columns
            col1, col2 = st.columns(2)
            
            # Display uploaded image
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Make prediction
            with col2:
                st.subheader("Prediction Results")
                
                # Get predictions
                predictions = predict(model, image)
                
                if predictions:
                    # Get top prediction
                    top_class = max(predictions.items(), key=lambda x: x[1])
                    
                    # Display results
                    st.markdown(f"### Predicted Class: **{top_class[0].title()}**")
                    st.markdown(f"### Confidence: **{top_class[1]:.2%}**")
                    
                    # Create DataFrame for visualization
                    df = pd.DataFrame(
                        list(predictions.items()),
                        columns=['Class', 'Probability']
                    )
                    
                    # Sort values
                    df = df.sort_values('Probability', ascending=True)
                    
                    # Show probabilities
                    st.subheader("Class Probabilities")
                    st.bar_chart(df.set_index('Class'))
                    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
