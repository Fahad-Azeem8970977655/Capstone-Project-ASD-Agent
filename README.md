ASD Early Screening AI Assistant


🧠 Project Overview


A comprehensive machine learning system for early Autism Spectrum Disorder screening using 15 behavioral indicators. Provides both web interface and API for ASD risk assessment through a trained Random Forest classifier with 85% accuracy.

⚡ Quick Start
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Launch web app
streamlit run streamlit_app.py

# Start API server


python agent_api.py


🎯 Key Features


15-question screening based on established behavioral markers

Real-time risk assessment with probability scoring

Interactive web interface with progress tracking

REST API for developers

Multiple risk categories with color-coded results

Export capabilities (CSV, PDF reports)

🏗️ System Architecture


Backend: FastAPI with machine learning model


Frontend: Streamlit web application


Model: Random Forest classifier (200 estimators)

Data: Comprehensive preprocessing pipeline

📊 Screening Areas


Social Communication: Response to name, eye contact, pointing

Social Interaction: Pretend play, empathy, gestures

Sensory & Development: Sound sensitivities, language milestones

Behavior Patterns: Routine adherence, social engagement

🔧 API Endpoints


POST /predict - Risk assessment

GET /health - System status

GET /model-info - Technical details

GET /features - Behavioral indicators

🚀 Usage


Web App: Access at http://localhost:8501

API: Available at http://localhost:8000

Integration: JSON API for custom applications

Results: Probability scores + risk categories + explanations

⚠️ Important Disclaimer

This is a screening tool only - not a diagnostic tool. Always consult qualified healthcare professionals for medical diagnosis. Results should be used as part of comprehensive evaluation by developmental specialists.

🤝 Support & Contribution


Issues: GitHub issue tracking

Enhancements: ML improvements, multi-language support

Medical Questions: Consult healthcare providers

Professional medical evaluation required for diagnosis. Use this tool for awareness and preliminary screening only.
