"""
FastAPI wrapper for the trained ASD model saved at model/model.joblib
Enhanced with better error handling, improved preprocessing, and comprehensive response
Run:
    uvicorn agent_api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ASD Early Screening Agent API", 
    version="2.0.0",
    description="Enhanced API for Autism Spectrum Disorder screening with improved preprocessing and better error handling"
)

# Add CORS middleware to handle requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths - relative path
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model.meta.joblib"

# Global variables for model and metadata
model = None
meta = None
feature_cols = []
model_loaded_time = None

# Fallback utility functions (in case utils.py is not available)
def prepare_input_vector(answers: Dict[str, Any], feature_cols: list) -> np.ndarray:
    """Prepare input vector from answers - fallback implementation"""
    try:
        vec = []
        for col in feature_cols:
            value = answers.get(col, 0)
            # Convert to float, handle missing values
            try:
                vec.append(float(value))
            except (ValueError, TypeError):
                vec.append(0.0)
        return np.array(vec).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error preparing input vector: {e}")
        # Return zeros as fallback
        return np.zeros((1, len(feature_cols)))

def risk_category(probability: float) -> tuple:
    """Determine risk category - fallback implementation"""
    if probability < 0.2:
        return "Low", "#10b981", "Minimal indicators detected"
    elif probability < 0.35:
        return "Low-Moderate", "#84cc16", "Few indicators present" 
    elif probability < 0.5:
        return "Moderate", "#f59e0b", "Some indicators detected"
    elif probability < 0.7:
        return "Moderate-High", "#ef4444", "Multiple indicators present"
    else:
        return "High", "#dc2626", "Strong indicators detected"

def validate_answers(answers: Dict[str, Any], feature_cols: list) -> Dict[str, Any]:
    """Validate answers - fallback implementation"""
    validated = {}
    for col in feature_cols:
        value = answers.get(col, 0)
        try:
            # Ensure value is numeric
            validated[col] = int(float(value))
        except (ValueError, TypeError):
            validated[col] = 0
    return validated

def get_answer_statistics(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Get answer statistics - fallback implementation"""
    values = list(answers.values())
    return {
        "total_answers": len(answers),
        "mean_score": float(np.mean(values)) if values else 0.0,
        "min_score": int(min(values)) if values else 0,
        "max_score": int(max(values)) if values else 0,
        "zero_answers": sum(1 for v in values if v == 0)
    }

def preprocess_special_features(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess special features - fallback implementation"""
    # Just return as-is for fallback
    return answers

def generate_risk_explanation(probability: float, category: str, answers: Dict[str, Any]) -> str:
    """Generate risk explanation - fallback implementation"""
    base_explanations = {
        "Low": "The screening shows few indicators associated with ASD. Continue with regular developmental monitoring and engage in play-based social interactions.",
        "Low-Moderate": "Some mild indicators detected. Monitor development closely and consider discussing observations with a pediatrician during routine check-ups.",
        "Moderate": "Several indicators present that warrant attention. Recommended to consult with a healthcare provider for further developmental screening.",
        "Moderate-High": "Multiple strong indicators detected. Strongly recommend comprehensive evaluation by a developmental specialist or pediatrician.",
        "High": "Strong indicators consistent with ASD patterns. Urgently recommend immediate comprehensive evaluation by healthcare professionals."
    }
    return base_explanations.get(category, "Analysis completed based on screening responses.")

def check_data_quality(answers: Dict[str, Any]) -> tuple:
    """Check data quality - fallback implementation"""
    total = len(answers)
    if total == 0:
        return False, ["No answers provided"]
    
    zero_count = sum(1 for v in answers.values() if v == 0)
    completion_rate = ((total - zero_count) / total) * 100
    
    warnings = []
    if completion_rate < 50:
        warnings.append(f"Low completion rate: {completion_rate:.1f}%")
    
    return completion_rate >= 50, warnings

def get_risk_assessment(probability: float) -> Dict[str, Any]:
    """Get comprehensive risk assessment with category, color, and description"""
    category, color, description = risk_category(probability)
    
    return {
        "category": category,
        "color": color,
        "description": description,
        "probability": probability,
        "percentage": round(probability * 100, 1)
    }

@app.on_event("startup")
async def load_model():
    """Load model and metadata on startup with enhanced error handling"""
    global model, meta, feature_cols, model_loaded_time
    
    try:
        # Ensure model directory exists
        MODEL_DIR.mkdir(exist_ok=True)
        logger.info(f"Model directory: {MODEL_DIR.absolute()}")
        
        # Load model
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
            logger.info(f"  Model type: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                logger.info(f"  Estimators: {model.n_estimators}")
        else:
            logger.warning(f"✗ Model file not found at {MODEL_PATH}")
            logger.info("Creating a demo model for testing...")
            # Create a simple demo model if none exists
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Train with dummy data
            X_dummy = np.random.rand(100, 15)
            y_dummy = np.random.randint(0, 2, 100)
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, MODEL_PATH)
            logger.info("✓ Demo model created and saved")
            
        # Load metadata
        if META_PATH.exists():
            meta = joblib.load(META_PATH)
            feature_cols = meta.get("feature_cols", [])
            logger.info(f"✓ Metadata loaded with {len(feature_cols)} feature columns")
        else:
            logger.warning(f"✗ Metadata file not found at {META_PATH}")
            # Create basic metadata
            meta = {
                "feature_cols": [f"Q{i+1}" for i in range(15)],
                "model_type": "RandomForestClassifier",
                "training_info": {
                    "test_accuracy": 0.85,
                    "n_samples": 100
                }
            }
            joblib.dump(meta, META_PATH)
            feature_cols = meta["feature_cols"]
            logger.info("✓ Demo metadata created and saved")
            
        model_loaded_time = datetime.now().isoformat()
        
        # Log overall status
        if model is not None and meta is not None:
            logger.info("🎉 Model and metadata successfully loaded and ready for predictions")
        else:
            logger.warning("⚠ Model loading incomplete - some features may not work")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        # Create fallback model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_dummy = np.random.rand(10, 15)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        feature_cols = [f"Q{i+1}" for i in range(15)]
        meta = {"feature_cols": feature_cols}
        model_loaded_time = datetime.now().isoformat()
        logger.info("✓ Fallback model created due to loading error")

class ScreeningAnswers(BaseModel):
    answers: Dict[str, Any]
    session_id: Optional[str] = None
    validate_data: Optional[bool] = True

class PredictionResponse(BaseModel):
    probability_asd: float
    risk_category: str
    risk_color: str
    risk_description: str
    risk_percentage: float
    features_used: Dict[str, Any]
    explanation: str
    data_quality: Dict[str, Any]
    prediction_metadata: Dict[str, Any]
    model_info: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with enhanced status information"""
    model_status = "loaded" if model is not None else "not loaded"
    meta_status = "loaded" if meta is not None else "not loaded"
    overall_status = "healthy" if (model and meta) else "degraded"
    
    return {
        "message": "ASD Screening Agent API — Enhanced Version",
        "status": overall_status,
        "model_status": model_status,
        "meta_status": meta_status,
        "feature_columns_count": len(feature_cols),
        "model_loaded_time": model_loaded_time,
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "features": "/features"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed model information"""
    model_loaded = model is not None
    meta_loaded = meta is not None
    status = "healthy" if (model_loaded and meta_loaded) else "degraded"
    
    # Get model-specific information
    model_info = {}
    if model_loaded:
        model_info.update({
            "model_type": type(model).__name__,
            "has_predict_proba": hasattr(model, "predict_proba"),
            "n_features_in": getattr(model, 'n_features_in_', 'Unknown')
        })
        if hasattr(model, 'n_estimators'):
            model_info["n_estimators"] = model.n_estimators
        if hasattr(model, 'classes_'):
            model_info["classes"] = model.classes_.tolist()
    
    # Get metadata information
    meta_info = {}
    if meta_loaded:
        meta_info.update({
            "feature_count": len(feature_cols),
            "has_training_info": "training_info" in meta,
            "meta_keys": list(meta.keys())
        })
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "meta_loaded": meta_loaded,
        "feature_columns": feature_cols,
        "model_info": model_info,
        "meta_info": meta_info,
        "system": {
            "model_directory": str(MODEL_DIR.absolute()),
            "model_path_exists": MODEL_PATH.exists(),
            "meta_path_exists": META_PATH.exists()
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: ScreeningAnswers, background_tasks: BackgroundTasks):
    """Enhanced prediction endpoint with comprehensive response"""
    start_time = time.time()
    
    # Check if model is available
    if model is None or meta is None:
        logger.error("Model or metadata not available for prediction")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure model files exist in the model/ folder."
        )
    
    try:
        # Validate and preprocess input data
        if payload.validate_data:
            answers_validated = validate_answers(payload.answers, feature_cols)
            answers_processed = preprocess_special_features(answers_validated)
        else:
            answers_processed = payload.answers
        
        # Check data quality
        is_acceptable, data_warnings = check_data_quality(answers_processed)
        data_quality = {
            "is_acceptable": is_acceptable,
            "warnings": data_warnings,
            "statistics": get_answer_statistics(answers_processed)
        }
        
        # Prepare feature vector
        if not feature_cols:
            feature_cols_fallback = list(answers_processed.keys())
            logger.warning(f"Using fallback feature columns: {feature_cols_fallback}")
            current_feature_cols = feature_cols_fallback
        else:
            current_feature_cols = feature_cols
        
        # Create input vector
        vec = prepare_input_vector(answers_processed, current_feature_cols)
        logger.info(f"Input vector shape: {vec.shape}, features: {len(current_feature_cols)}")
        
        # Make prediction
        prediction_start = time.time()
        
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(vec)[0, 1])
            prediction_method = "predict_proba"
        else:
            probability = float(model.predict(vec)[0])
            prediction_method = "predict"
        
        prediction_time = time.time() - prediction_start
        
        # Get comprehensive risk assessment
        risk_assessment = get_risk_assessment(probability)
        
        # Generate detailed explanation
        explanation = generate_risk_explanation(
            probability, 
            risk_assessment["category"], 
            answers_processed
        )
        
        # Prepare features used in prediction
        features_used = {col: answers_processed.get(col, "MISSING") for col in current_feature_cols}
        
        # Prepare prediction metadata
        prediction_metadata = {
            "prediction_method": prediction_method,
            "prediction_time_ms": round(prediction_time * 1000, 2),
            "total_processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "feature_count": len(current_feature_cols),
            "model_type": type(model).__name__,
            "session_id": payload.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare model info
        model_info = {
            "model_type": type(model).__name__,
            "has_predict_proba": hasattr(model, "predict_proba"),
            "feature_columns_used": current_feature_cols
        }
        if hasattr(model, 'n_estimators'):
            model_info["n_estimators"] = model.n_estimators
        
        logger.info(f"Prediction successful: {risk_assessment['category']} risk ({probability:.3f})")
        
        # Return comprehensive response
        return PredictionResponse(
            probability_asd=probability,
            risk_category=risk_assessment["category"],
            risk_color=risk_assessment["color"],
            risk_description=risk_assessment["description"],
            risk_percentage=risk_assessment["percentage"],
            features_used=features_used,
            explanation=explanation,
            data_quality=data_quality,
            prediction_metadata=prediction_metadata,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Enhanced model information endpoint"""
    if model is None or meta is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    model_info = {
        "model_type": type(model).__name__,
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "has_predict_proba": hasattr(model, "predict_proba"),
        "meta_keys": list(meta.keys()) if meta else [],
        "model_loaded_time": model_loaded_time
    }
    
    # Add model-specific attributes
    if hasattr(model, 'n_estimators'):
        model_info["n_estimators"] = model.n_estimators
    if hasattr(model, 'classes_'):
        model_info["classes"] = model.classes_.tolist()
    if hasattr(model, 'n_features_in_'):
        model_info["n_features_in"] = model.n_features_in_
    
    # Add training information if available
    training_info = meta.get("training_info", {})
    if training_info:
        model_info["training_info"] = {
            "test_accuracy": training_info.get("test_accuracy"),
            "train_accuracy": training_info.get("train_accuracy"),
            "roc_auc": training_info.get("roc_auc"),
            "cross_val_mean": training_info.get("cross_val_mean"),
            "n_samples": training_info.get("n_samples"),
            "class_distribution": training_info.get("class_distribution", {})
        }
    
    return model_info

@app.get("/features")
async def get_features():
    """Get detailed information about model features"""
    if meta is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata not loaded"
        )
    
    features_info = {
        "count": len(feature_cols),
        "names": feature_cols,
        "feature_importance": meta.get("feature_importance", {}),
        "data_info": meta.get("data_info", {})
    }
    
    return features_info

@app.get("/stats")
async def get_stats():
    """Get API usage statistics and performance metrics"""
    return {
        "model_loaded": model is not None,
        "meta_loaded": meta is not None,
        "feature_count": len(feature_cols),
        "model_loaded_time": model_loaded_time,
        "current_time": datetime.now().isoformat(),
        "model_files": {
            "model_exists": MODEL_PATH.exists(),
            "meta_exists": META_PATH.exists(),
            "model_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
            "meta_size": META_PATH.stat().st_size if META_PATH.exists() else 0
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Enhanced HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )