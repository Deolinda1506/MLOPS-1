from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

# Import our modules
from preprocessing import load_data, LungCancerDataPreprocessor
from model import build_model, train_model
from prediction import load_trained_model, predict_image, get_lung_cancer_class_labels
from database import db

app = FastAPI(
    title="Lung Cancer Classification API",
    description="API for lung cancer classification using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables
MODEL_PATH = "models/best_model.h5"
TRAINING_STATUS = {"status": "idle", "progress": 0, "message": ""}
PREDICTION_HISTORY = []

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Cancer Classification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 5px; }
            .upload-area:hover { border-color: #007bff; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Lung Cancer Classification</h1>
            <p>Upload a lung X-ray image to classify it into one of the following categories:</p>
            <ul>
                <li><strong>Adenocarcinoma</strong></li>
                <li><strong>Large Cell Carcinoma</strong></li>
                <li><strong>Squamous Cell Carcinoma</strong></li>
                <li><strong>Normal</strong></li>
            </ul>
            
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit" class="btn">🔍 Analyze Image</button>
                </div>
            </form>
            
            <div>
                <h3>📊 API Endpoints:</h3>
                <ul>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>POST /predict</code> - Single image prediction</li>
                    <li><code>POST /predict-batch</code> - Batch prediction</li>
                    <li><code>POST /retrain</code> - Retrain model</li>
                    <li><code>GET /status</code> - Training status</li>
                    <li><code>GET /history</code> - Prediction history</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": os.path.exists(MODEL_PATH)
    }

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    """Predict single image"""
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
        
        model = load_trained_model(MODEL_PATH)
        class_labels = get_lung_cancer_class_labels()
        
        # Make prediction
        predicted_label, confidence = predict_image(model, file_path, class_labels)
        
        # Save to database
        db.save_prediction(
            filename=file.filename,
            predicted_class=predicted_label,
            confidence=confidence
        )
        
        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
            "timestamp": prediction_record["timestamp"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch_images(files: list[UploadFile] = File(...)):
    """Predict multiple images"""
    try:
        results = []
        
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
        
        model = load_trained_model(MODEL_PATH)
        class_labels = get_lung_cancer_class_labels()
        
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
                
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Make prediction
            predicted_label, confidence = predict_image(model, file_path, class_labels)
            
            results.append({
                "filename": file.filename,
                "prediction": predicted_label,
                "confidence": confidence
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the model with new data"""
    try:
        if TRAINING_STATUS["status"] == "training":
            return {"message": "Training already in progress"}
        
        # Create training session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.create_training_session(session_id, "data/")
        
        background_tasks.add_task(train_model_background, session_id)
        return {"message": "Retraining started in background", "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_training_data(files: list[UploadFile] = File(...)):
    """Upload training data for retraining"""
    try:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        uploaded_files = []
        
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
                
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Save to database
            db.save_uploaded_data(
                filename=file.filename,
                file_path=file_path,
                file_size=len(file.file.read()),
                session_id=session_id
            )
            
            uploaded_files.append(file.filename)
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files",
            "session_id": session_id,
            "files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(session_id=None):
    """Background task for model training"""
    global TRAINING_STATUS
    
    try:
        TRAINING_STATUS = {"status": "training", "progress": 0, "message": "Loading data..."}
        
        # Load data
        train_dir = "data/train"
        val_dir = "data/valid"
        test_dir = "data/test"
        
        train_gen, val_gen, test_gen, class_indices = load_data(
            train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=32
        )
        
        TRAINING_STATUS["progress"] = 20
        TRAINING_STATUS["message"] = "Building model..."
        
        # Build and train model
        input_shape = (224, 224, 3)
        num_classes = len(class_indices)
        
        model, history = train_model(
            train_gen, val_gen, input_shape, num_classes,
            model_path=MODEL_PATH
        )
        
        # Update database with training results
        if session_id:
            final_accuracy = history.history['accuracy'][-1]
            final_loss = history.history['loss'][-1]
            epochs = len(history.history['accuracy'])
            
            db.update_training_session(
                session_id,
                end_time=datetime.now(),
                status="completed",
                epochs=epochs,
                final_accuracy=final_accuracy,
                final_loss=final_loss,
                model_path=MODEL_PATH
            )
            
            # Save model version
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db.save_model_version(version, MODEL_PATH, final_accuracy, final_loss, session_id)
        
        TRAINING_STATUS = {"status": "completed", "progress": 100, "message": "Training completed successfully"}
        
    except Exception as e:
        if session_id:
            db.update_training_session(session_id, status="error", end_time=datetime.now())
        TRAINING_STATUS = {"status": "error", "progress": 0, "message": str(e)}

@app.get("/status")
async def get_training_status():
    """Get training status"""
    return TRAINING_STATUS

@app.get("/history")
async def get_prediction_history():
    """Get prediction history"""
    # Get from database instead of memory
    db_history = db.get_prediction_history(50)
    return {"history": db_history}

@app.get("/training-sessions")
async def get_training_sessions():
    """Get training sessions history"""
    sessions = db.get_training_sessions(10)
    return {"sessions": sessions}

@app.get("/uploaded-data/{session_id}")
async def get_uploaded_data(session_id: str):
    """Get uploaded data for a training session"""
    data = db.get_uploaded_data_for_session(session_id)
    return {"session_id": session_id, "data": data}

@app.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        if not os.path.exists("models/history.json"):
            return {"message": "No training history available"}
        
        with open("models/history.json", "r") as f:
            history = json.load(f)
        
        # Get final metrics
        final_metrics = {
            "final_train_accuracy": history.get("accuracy", [0])[-1],
            "final_val_accuracy": history.get("val_accuracy", [0])[-1],
            "final_train_loss": history.get("loss", [0])[-1],
            "final_val_loss": history.get("val_loss", [0])[-1],
            "training_epochs": len(history.get("accuracy", []))
        }
        
        return final_metrics
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 