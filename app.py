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
    """Create VGG16-based model"""
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def try_load_model(model_path):
    """Try different methods to load the model"""
    st.write("Attempting to load model...")
    
    try:
        # Method 1: Try loading as SavedModel
        st.write("Trying to load as SavedModel...")
        if os.path.isdir(model_path.replace('.keras', '')):
            model = tf.keras.models.load_model(model_path.replace('.keras', ''))
            st.success("Successfully loaded model as SavedModel!")
            return model
    except Exception as e:
        st.write(f"SavedModel loading failed: {str(e)}")

    try:
        # Method 2: Try loading as Keras model with custom objects
        st.write("Trying to load as Keras model with custom objects...")
        custom_objects = {
            'CustomLoss': tf.keras.losses.CategoricalCrossentropy,
            'CustomMetric': tf.keras.metrics.CategoricalAccuracy
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Successfully loaded model with custom objects!")
        return model
    except Exception as e:
        st.write(f"Keras model loading failed: {str(e)}")

    try:
        # Method 3: Try loading just weights
        st.write("Trying to load weights into fresh model...")
        model = create_scene_model()
        model.load_weights(model_path)
        st.success("Successfully loaded weights into fresh model!")
        return model
    except Exception as e:
        st.write(f"Weight loading failed: {str(e)}")
    
    try:
        # Method 4: Try loading weights from HDF5 if exists
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            st.write("Trying to load HDF5 weights...")
            model = create_scene_model()
            model.load_weights(h5_path)
            st.success("Successfully loaded HDF5 weights!")
            return model
    except Exception as e:
        st.write(f"HDF5 loading failed: {str(e)}")
    
    # If all attempts fail, create new model with ImageNet weights
    st.warning("All loading attempts failed. Creating new model with ImageNet weights.")
    return create_scene_model()

@st.cache_resource
def load_classification_model():
    """Load or create the model with multiple fallback options"""
    try:
        model_path = 'best_vgg16.keras'
        
        # Try different loading methods
        model = try_load_model(model_path)
        
        if model is None:
            st.error("Failed to initialize model")
            return None
            
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
        
        # Normalize
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
    
    # Show debug info in sidebar
    with st.sidebar:
        st.subheader("Debug Information")
        st.write(f"TensorFlow version: {tf.__version__}")
        st.write(f"Keras version: {tf.keras.__version__}")
    
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
                    probabilities_df = pd.DataFrame({
                        'Class': list(class_probabilities.keys()),
                        'Probability': list(class_probabilities.values())
                    })
                    probabilities_df['Class'] = probabilities_df['Class'].str.title()
                    probabilities_df = probabilities_df.sort_values('Probability', ascending=True)
                    
                    st.bar_chart(
                        probabilities_df.set_index('Class'),
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the image format.")

if __name__ == "__main__":
    main()
