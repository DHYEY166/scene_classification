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

# Set page configuration
st.set_page_config(
    page_title="Scene Classification App",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for essential styling
st.markdown("""
<style>
    .main h1 {
        color: inherit;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem;
    }
    
    .stSpinner > div {
        border-top-color: #2563EB !important;
    }
    
    .prediction-container {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 0.5rem;
    }
    
    /* Ensure adequate spacing */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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
        model = create_model()
        
        weights_path = 'scene_classifier.weights.h5'
        if not os.path.exists(weights_path):
            st.error(f"Weights file not found at: {weights_path}")
            return None
        
        model.load_weights(weights_path)
        
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
    """Preprocess image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
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
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None
        
        predictions = model.predict(processed_img, verbose=0)
        
        results = {
            class_name: float(prob)
            for class_name, prob in zip(classes, predictions[0])
        }
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Page title
    st.markdown("<h1>🏞️ Scene Classification AI</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 About")
        st.info("""
        Welcome to Scene Classification AI!
        
        This application uses advanced deep learning to classify scenes 
        into six different categories. Perfect for photographers, researchers, 
        and nature enthusiasts.
        
        Upload any scene image to get instant classification results!
        """)
        
        # Model details in expander
        with st.expander("🔍 Technical Details"):
            st.info("""
            **Model Architecture:**
            * Base: VGG16
            * Input Size: 224x224 pixels
            * Categories: 6 scene types
            * Technology: Transfer Learning
            * Backend: TensorFlow 2.x
            
            **Supported Categories:**
            * 🏢 Buildings
            * 🌲 Forest
            * ❄️ Glacier
            * ⛰️ Mountain
            * 🌊 Sea
            * 🛣️ Street
            """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a scene image to classify",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        For best results:
        * Use clear, well-lit images
        * Avoid blurry photos
        * Center the main scene
        * Landscape orientation preferred
        """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    else:
        st.success("Model loaded successfully!")
    
    # Process uploaded image
    if uploaded_file is not None:
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📸 Uploaded Scene")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col4:
            st.markdown("### 🎯 Analysis Results")
            
            with st.spinner('🔄 Analyzing scene...'):
                predictions = predict_scene(model, image)
            
            if predictions:
                top_class = max(predictions.items(), key=lambda x: x[1])
                
                # Use st.success for the prediction box
                st.success(f"""
                🎪 **Predicted Scene:** {top_class[0].title()}
                📊 **Confidence:** {top_class[1]:.1%}
                """)
                
                # Show all probabilities
                st.markdown("### 📈 Detailed Analysis")
                
                df = pd.DataFrame(
                    list(predictions.items()),
                    columns=['Class', 'Probability']
                )
                df = df.sort_values('Probability', ascending=True)
                
                # Use Streamlit's native chart with custom height
                st.bar_chart(
                    df.set_index('Class'),
                    use_container_width=True,
                    height=400
                )

if __name__ == "__main__":
    main()
