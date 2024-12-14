import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Scene Classification App",
    page_icon="🌄",
    layout="wide"
)

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def create_scene_model():
    """Create VGG16-based model with proper initialization"""
    # Load VGG16 with pretrained weights
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Build the complete model
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        
        # VGG16 base
        base_model
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_resource
def load_classification_model():
    """Load or create the model"""
    try:
        # Create base model
        model = create_scene_model()
        
        # Load custom weights if available
        weights_path = 'best_vgg16.keras'
        if os.path.exists(weights_path):
            try:
                # Try loading weights
                model.load_weights(weights_path)
                st.success("Custom weights loaded successfully!")
            except Exception as e:
                st.warning(f"Could not load custom weights: {str(e)}")
                st.info("Using base model with ImageNet weights.")
        else:
            st.warning("Model weights file not found. Using base model with ImageNet weights.")
        
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(image, dtype=np.float32)
        
        # Preprocess input (scale to [0, 1])
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Predict scene category from image"""
    try:
        # Preprocess image
        img_array = preprocess_image(image)
        if img_array is None:
            return None, None, None
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        # Get all class probabilities
        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
        
        return predicted_class, confidence, class_probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Title and description
    st.title("Scene Classification using VGG16")
    st.write("Upload an image of a scene to classify it into one of six categories: buildings, forest, glacier, mountain, sea, or street.")
    
    # Display version info in sidebar
    st.sidebar.write(f"TensorFlow version: {tf.__version__}")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_classification_model()
    
    if model is None:
        st.error("Failed to initialize the model.")
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
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, class_probabilities = predict_scene(model, image)
            
            if predicted_class is not None:
                # Display results
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Show prediction with confidence
                    st.markdown(
                        f"""
                        <div style='padding: 20px; border-radius: 5px; background-color: #f0f2f6;'>
                            <h3>Predicted Scene: {predicted_class.title()}</h3>
                            <h4>Confidence: {confidence:.1%}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display probability bar chart
                    st.subheader("Class Probabilities")
                    
                    # Create and sort DataFrame
                    probabilities_df = pd.DataFrame({
                        'Class': list(class_probabilities.keys()),
                        'Probability': list(class_probabilities.values())
                    })
                    probabilities_df['Class'] = probabilities_df['Class'].str.title()
                    probabilities_df = probabilities_df.sort_values('Probability', ascending=True)
                    
                    # Create bar chart
                    st.bar_chart(
                        probabilities_df.set_index('Class'),
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the image format.")

if __name__ == "__main__":
    main()
