#!/usr/bin/env python3
"""
Streamlit Web Interface for Lung Cancer Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")  # Will be set in Streamlit Cloud

# Import local modules for training (only if running locally)
try:
    from preprocessing import load_data, LungCancerDataPreprocessor
    from model import build_model, train_model
    from prediction import load_trained_model, predict_image, get_lung_cancer_class_labels
    LOCAL_MODE = True
except ImportError:
    LOCAL_MODE = False
    st.warning("⚠️ Running in deployment mode - using API for predictions")

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-confidence {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .medium-confidence {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    .low-confidence {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Lung Cancer Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "🤖 Model Training", "📈 Analytics", "📋 About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Single Prediction":
        show_single_prediction()
    elif page == "📊 Batch Prediction":
        show_batch_prediction()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "📋 About":
        show_about_page()

def show_home_page():
    """Home page with overview and quick actions"""
    st.header("Welcome to Lung Cancer Classification System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What We Do")
        st.write("""
        This system uses deep learning to classify lung X-ray images into four categories:
        
        - **Adenocarcinoma** - A type of non-small cell lung cancer
        - **Large Cell Carcinoma** - Another type of non-small cell lung cancer  
        - **Squamous Cell Carcinoma** - A type of non-small cell lung cancer
        - **Normal** - Healthy lung tissue
        
        Our model achieves high accuracy in detecting these different types of lung cancer.
        """)
        
        st.subheader("🚀 Quick Actions")
        if st.button("🔍 Try Single Prediction"):
            st.switch_page("🔍 Single Prediction")
        if st.button("📊 Batch Analysis"):
            st.switch_page("📊 Batch Prediction")
        if st.button("🤖 Train Model"):
            st.switch_page("🤖 Model Training")
    
    with col2:
        st.subheader("📊 System Status")
        
        # Check if model exists
        model_exists = os.path.exists("models/best_model.h5")
        
        if model_exists:
            st.success("✅ Model is ready for predictions")
            
            # Load model info
            try:
                model = load_trained_model("models/best_model.h5")
                st.metric("Model Parameters", f"{model.count_params():,}")
            except:
                st.info("Model loaded but details unavailable")
        else:
            st.warning("⚠️ Model not found. Please train the model first.")
        
        # Show recent activity
        st.subheader("📈 Recent Activity")
        if os.path.exists("models/history.json"):
            try:
                with open("models/history.json", "r") as f:
                    history = json.load(f)
                
                if "accuracy" in history:
                    final_acc = history["accuracy"][-1]
                    st.metric("Final Training Accuracy", f"{final_acc:.2%}")
            except:
                st.info("No training history available")
        else:
            st.info("No training history available")

def show_single_prediction():
    """Single image prediction page"""
    st.header("🔍 Single Image Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose a lung X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a lung X-ray image for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            if LOCAL_MODE:
                # Local mode - use local model
                if not os.path.exists("models/best_model.h5"):
                    st.error("❌ Model not found. Please train the model first.")
                    return
                
                try:
                    # Load model and make prediction
                    model = load_trained_model("models/best_model.h5")
                    class_labels = get_lung_cancer_class_labels()
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Make prediction
                    predicted_label, confidence = predict_image(model, temp_path, class_labels)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    display_prediction_results(predicted_label, confidence)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
            else:
                # Deployment mode - use API
                try:
                    import requests
                    
                    # Prepare file for API
                    files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "image/png")}
                    
                    # Make API call
                    response = requests.post(f"{API_BASE_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_label = result["prediction"]
                        confidence = result["confidence"]
                        
                        display_prediction_results(predicted_label, confidence)
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error connecting to API: {str(e)}")

def display_prediction_results(predicted_label, confidence):
    """Display prediction results with styling"""
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    # Confidence indicator
    if confidence > 0.8:
        confidence_class = "high-confidence"
        confidence_emoji = "🟢"
    elif confidence > 0.6:
        confidence_class = "medium-confidence"
        confidence_emoji = "🟡"
    else:
        confidence_class = "low-confidence"
        confidence_emoji = "🔴"
    
    st.markdown(f"""
    <div class="prediction-result {confidence_class}">
        <h4>{confidence_emoji} Confidence Level: {confidence:.1%}</h4>
        <p><strong>Prediction:</strong> {predicted_label}</p>
    </div>
    """, unsafe_allow_html=True)

def show_batch_prediction():
    """Batch prediction page"""
    st.header("📊 Batch Prediction")
    
    uploaded_files = st.file_uploader(
        "Choose multiple lung X-ray images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple lung X-ray images for batch classification"
    )
    
    if uploaded_files:
        st.write(f"📁 Uploaded {len(uploaded_files)} images")
        
        # Check if model exists
        if not os.path.exists("models/best_model.h5"):
            st.error("❌ Model not found. Please train the model first.")
            return
        
        if st.button("🔍 Analyze All Images"):
            with st.spinner("Analyzing images..."):
                try:
                    # Load model
                    model = load_trained_model("models/best_model.h5")
                    class_labels = get_lung_cancer_class_labels()
                    
                    results = []
                    
                    # Process each image
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Make prediction
                        predicted_label, confidence = predict_image(model, temp_path, class_labels)
                        
                        results.append({
                            'Filename': uploaded_file.name,
                            'Prediction': predicted_label,
                            'Confidence': confidence,
                            'Confidence_Percent': f"{confidence:.1%}"
                        })
                        
                        # Clean up
                        os.remove(temp_path)
                    
                    # Display results
                    st.subheader("📊 Batch Analysis Results")
                    
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Prediction Distribution")
                        pred_counts = df['Prediction'].value_counts()
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                                   title="Predictions by Class")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Confidence Distribution")
                        fig = px.histogram(df, x='Confidence', nbins=20,
                                         title="Confidence Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during batch analysis: {str(e)}")

def show_model_training():
    """Model training page"""
    st.header("🤖 Model Training")
    
    # Check current model status
    model_exists = os.path.exists("models/best_model.h5")
    
    if model_exists:
        st.success("✅ Model exists and is ready for use")
        
        # Show model info
        try:
            model = load_trained_model("models/best_model.h5")
            st.metric("Model Parameters", f"{model.count_params():,}")
        except:
            st.info("Model loaded but details unavailable")
    else:
        st.warning("⚠️ No trained model found")
    
    # Training options
    st.subheader("🔄 Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training Configuration")
        
        epochs = st.slider("Number of Epochs", 5, 50, 20)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.01], index=0)
        
        use_class_weights = st.checkbox("Use Class Weights", value=True, 
                                      help="Balance imbalanced dataset")
    
    with col2:
        st.subheader("📁 Data Information")
        
        # Check data availability
        data_dirs = ["data/train", "data/valid", "data/test"]
        data_status = {}
        
        for dir_path in data_dirs:
            if os.path.exists(dir_path):
                try:
                    num_files = len([f for f in os.listdir(dir_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    data_status[dir_path] = num_files
                except:
                    data_status[dir_path] = "Error"
            else:
                data_status[dir_path] = "Not found"
        
        for dir_path, status in data_status.items():
            if status == "Not found":
                st.error(f"❌ {dir_path}: {status}")
            elif status == "Error":
                st.warning(f"⚠️ {dir_path}: {status}")
            else:
                st.success(f"✅ {dir_path}: {status} images")
    
    # Start training
    if st.button("🚀 Start Training", type="primary"):
        if not all(os.path.exists(d) for d in data_dirs):
            st.error("❌ Please ensure all data directories exist")
            return
        
        with st.spinner("Training model..."):
            try:
                # Load data
                train_gen, val_gen, test_gen, class_indices = load_data(
                    "data/train", "data/valid", "data/test",
                    image_size=(224, 224), batch_size=batch_size
                )
                
                # Calculate class weights if requested
                class_weight_dict = None
                if use_class_weights:
                    from sklearn.utils.class_weight import compute_class_weight
                    labels_numerical = train_gen.classes
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=np.unique(labels_numerical), 
                        y=labels_numerical
                    )
                    class_weight_dict = dict(enumerate(class_weights))
                
                # Build and train model
                input_shape = (224, 224, 3)
                num_classes = len(class_indices)
                
                model, history = train_model(
                    train_gen, val_gen, input_shape, num_classes,
                    class_weight=class_weight_dict
                )
                
                st.success("✅ Training completed successfully!")
                
                # Show training results
                st.subheader("📈 Training Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    final_acc = history.history['accuracy'][-1]
                    final_val_acc = history.history['val_accuracy'][-1]
                    st.metric("Final Training Accuracy", f"{final_acc:.2%}")
                    st.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")
                
                with col2:
                    final_loss = history.history['loss'][-1]
                    final_val_loss = history.history['val_loss'][-1]
                    st.metric("Final Training Loss", f"{final_loss:.4f}")
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                
                # Training plots
                st.subheader("📊 Training History")
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                axes[0].plot(history.history['accuracy'], label='Training')
                axes[0].plot(history.history['val_accuracy'], label='Validation')
                axes[0].set_title('Accuracy')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Accuracy')
                axes[0].legend()
                axes[0].grid(True)
                
                axes[1].plot(history.history['loss'], label='Training')
                axes[1].plot(history.history['val_loss'], label='Validation')
                axes[1].set_title('Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def show_analytics():
    """Analytics and model performance page"""
    st.header("📈 Analytics & Model Performance")
    
    # Check if training history exists
    if not os.path.exists("models/history.json"):
        st.warning("⚠️ No training history available. Please train the model first.")
        return
    
    try:
        with open("models/history.json", "r") as f:
            history = json.load(f)
        
        st.subheader("📊 Training Metrics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            final_acc = history.get("accuracy", [0])[-1]
            st.metric("Final Training Accuracy", f"{final_acc:.2%}")
        
        with col2:
            final_val_acc = history.get("val_accuracy", [0])[-1]
            st.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")
        
        with col3:
            final_loss = history.get("loss", [0])[-1]
            st.metric("Final Training Loss", f"{final_loss:.4f}")
        
        with col4:
            final_val_loss = history.get("val_loss", [0])[-1]
            st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
        
        # Training history plots
        st.subheader("📈 Training History")
        
        if "accuracy" in history:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=history["accuracy"],
                name="Training Accuracy",
                line=dict(color="blue")
            ))
            
            if "val_accuracy" in history:
                fig.add_trace(go.Scatter(
                    y=history["val_accuracy"],
                    name="Validation Accuracy",
                    line=dict(color="red")
                ))
            
            fig.update_layout(
                title="Model Accuracy Over Time",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if "loss" in history:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=history["loss"],
                name="Training Loss",
                line=dict(color="blue")
            ))
            
            if "val_loss" in history:
                fig.add_trace(go.Scatter(
                    y=history["val_loss"],
                    name="Validation Loss",
                    line=dict(color="red")
                ))
            
            fig.update_layout(
                title="Model Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture info
        if os.path.exists("models/best_model.h5"):
            st.subheader("🏗️ Model Architecture")
            try:
                model = load_trained_model("models/best_model.h5")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Parameters", f"{model.count_params():,}")
                    st.metric("Input Shape", str(model.input_shape[1:]))
                    st.metric("Output Shape", str(model.output_shape[1:]))
                
                with col2:
                    st.metric("Number of Layers", len(model.layers))
                    st.metric("Model Size", f"{os.path.getsize('models/best_model.h5') / 1024 / 1024:.1f} MB")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def show_about_page():
    """About page with project information"""
    st.header("📋 About This Project")
    
    st.subheader("🎯 Project Overview")
    st.write("""
    This Lung Cancer Classification System is a comprehensive machine learning pipeline 
    designed to classify lung X-ray images into different types of lung cancer and normal tissue.
    
    **Key Features:**
    - 🔬 Deep learning-based image classification
    - 📊 Real-time prediction capabilities
    - 🤖 Automated model training and retraining
    - 📈 Comprehensive analytics and visualization
    - 🌐 Web-based user interface
    """)
    
    st.subheader("🔬 Technical Details")
    st.write("""
    **Model Architecture:**
    - Convolutional Neural Network (CNN)
    - Transfer learning with pre-trained models
    - Data augmentation for improved generalization
    - Class balancing for imbalanced datasets
    
    **Technologies Used:**
    - TensorFlow/Keras for deep learning
    - Streamlit for web interface
    - FastAPI for REST API
    - OpenCV for image processing
    - Scikit-learn for metrics and utilities
    """)
    
    st.subheader("📊 Dataset Information")
    st.write("""
    The system is trained on a dataset of lung X-ray images with four classes:
    
    1. **Adenocarcinoma** - A type of non-small cell lung cancer
    2. **Large Cell Carcinoma** - Another type of non-small cell lung cancer
    3. **Squamous Cell Carcinoma** - A type of non-small cell lung cancer
    4. **Normal** - Healthy lung tissue
    
    The dataset includes training, validation, and test splits for proper model evaluation.
    """)
    
    st.subheader("🚀 Usage Instructions")
    st.write("""
    1. **Single Prediction**: Upload a single lung X-ray image for classification
    2. **Batch Prediction**: Upload multiple images for batch analysis
    3. **Model Training**: Train or retrain the model with your data
    4. **Analytics**: View model performance and training history
    
    **API Endpoints:**
    - `GET /health` - Health check
    - `POST /predict` - Single image prediction
    - `POST /predict-batch` - Batch prediction
    - `POST /retrain` - Retrain model
    - `GET /status` - Training status
    - `GET /metrics` - Model metrics
    """)
    
    st.subheader("📞 Contact & Support")
    st.write("""
    For questions, issues, or contributions, please refer to the project documentation
    or contact the development team.
    
    **Disclaimer:** This system is for educational and research purposes. 
    Medical decisions should always be made by qualified healthcare professionals.
    """)

if __name__ == "__main__":
    main() 