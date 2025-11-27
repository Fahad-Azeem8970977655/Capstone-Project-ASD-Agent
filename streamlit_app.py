"""
ASD Screening Streamlit App — Enhanced Version
- Updated for enhanced API compatibility
- Better error handling and user experience
- Improved result visualization
- Enhanced data validation
- Fixed session state initialization
- FIXED: Progress bar now works perfectly
"""

import streamlit as st
from pathlib import Path
import io, csv, requests, joblib, numpy as np, matplotlib.pyplot as plt, re
from datetime import datetime
import time
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
APP_TITLE = "ASD Early Screening — AI Assistant"
DEFAULT_API = "http://localhost:8000/predict"
MODEL_DIRS = [Path("model"), Path("src/model"), Path("./model")]  # Updated paths

# ---------------- Styling ----------------
CUSTOM_CSS = """
<style>
.block-container { max-width: 1100px; padding-top: 1rem; }
.card { background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 8px 28px rgba(13,38,63,0.06); border: 1px solid #eef2f7; }
.chat-bubble { border-radius: 14px; padding: 12px 16px; margin-bottom: 10px; line-height:1.4; display:block; }
.bot { background: #f3f4f6; color:#0b2442; float:left; max-width:85%; }
.user { background: #06b6d4; color:white; float:right; max-width:85%; }
.clearfix::after { content: ''; clear: both; display: table; }
.sep { height:1px; background:#eef2f7; margin:14px 0; border-radius:4px; }
.risk-pill { padding:8px 16px; border-radius:999px; font-weight:700; color:white; display:inline-block; margin: 5px 0; }
.small-muted { color:#6b7280; font-size:0.95rem; }
.warning-box { background: #fef3cd; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; margin: 10px 0; }
.success-box { background: #d1fae5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 12px; margin: 10px 0; }
.info-box { background: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 12px; margin: 10px 0; }
.progress-bar { height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
.progress-fill { height: 100%; background: #06b6d4; transition: width 0.3s ease; }
</style>
"""
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Questions ----------------
QUESTIONS = [
    "Q1. Does your child look at you when you call their name?",
    "Q2. Does your child point at things they want?",
    "Q3. Does your child point to share interest?",
    "Q4. Does your child look at your face while talking/playing?",
    "Q5. Does your child follow your gaze?",
    "Q6. Does your child engage in pretend play?",
    "Q7. If someone is upset, does your child show comfort attempts?",
    "Q8. Does your child get upset when routine changes?",
    "Q9. Does your child use simple gestures (wave, etc)?",
    "Q10. Did you ever suspect the child was nearly deaf?",
    "Q11. Does the child insist on specific music or sounds?",
    "Q12. In the first year, did the child react to lights/sounds?",
    "Q13. At what age did the child say first words?",
    "Q14. Approximately how many words can the child say?",
    "Q15. Hours per week playing with other children?",
]

# ---------------- Question-specific options ----------------
QUESTION_OPTIONS = {
    # Questions 1-11 use the standard 0-3 scale
    "Q1. Does your child look at you when you call their name?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Response to name calling"
    },
    "Q2. Does your child point at things they want?": {
        "type": "scale", 
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Requesting pointing"
    },
    "Q3. Does your child point to share interest?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Joint attention pointing"
    },
    "Q4. Does your child look at your face while talking/playing?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Eye contact during interaction"
    },
    "Q5. Does your child follow your gaze?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Gaze following"
    },
    "Q6. Does your child engage in pretend play?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Imaginative play"
    },
    "Q7. If someone is upset, does your child show comfort attempts?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Empathy and comfort"
    },
    "Q8. Does your child get upset when routine changes?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Response to routine changes"
    },
    "Q9. Does your child use simple gestures (wave, etc)?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Use of communicative gestures"
    },
    "Q10. Did you ever suspect the child was nearly deaf?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Hearing concerns"
    },
    "Q11. Does the child insist on specific music or sounds?": {
        "type": "scale",
        "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
        "description": "Sound sensitivities"
    },
    
    # Q12: Yes/No question
    "Q12. In the first year, did the child react to lights/sounds?": {
        "type": "yesno",
        "options": {0: "❌ No", 1: "✅ Yes"},
        "description": "Early sensory responses"
    },
    
    # Q13: Age categories
    "Q13. At what age did the child say first words?": {
        "type": "age",
        "options": {
            0: "👶 Before 9 months",
            1: "🧒 9-12 months", 
            2: "👦 1-2 years",
            3: "👨 After 2 years"
        },
        "description": "Language milestone"
    },
    
    # Q14: Word count ranges
    "Q14. Approximately how many words can the child say?": {
        "type": "words",
        "options": {
            0: "🗣️ 0-10 words",
            1: "🗣️ 11-30 words",
            2: "💬 31-50 words", 
            3: "🎯 50+ words"
        },
        "description": "Current vocabulary"
    },
    
    # Q15: Hours per week
    "Q15. Hours per week playing with other children?": {
        "type": "hours", 
        "options": {
            0: "🏠 0-2 hours",
            1: "👥 3-5 hours",
            2: "🤝 6-10 hours",
            3: "🌟 10+ hours"
        },
        "description": "Social interaction"
    }
}

# Set default options for questions not explicitly defined
for q in QUESTIONS:
    if q not in QUESTION_OPTIONS:
        QUESTION_OPTIONS[q] = {
            "type": "scale",
            "options": {0: "❌ Never", 1: "🙂 Sometimes", 2: "👍 Often", 3: "⭐ Very Often"},
            "description": "Behavior frequency"
        }

# ---------------- Enhanced Helpers ----------------
def sanitize_key(text: str) -> str:
    s = re.sub(r'\s+', '_', text.strip())
    return re.sub(r'[^A-Za-z0-9_]', '', s)

@st.cache_resource
def load_local_bundle():
    """Load local model & meta once (cached) with better error handling"""
    for d in MODEL_DIRS:
        m = d / "model.joblib"
        meta = d / "model.meta.joblib"
        if m.exists() and meta.exists():
            try:
                model = joblib.load(m)
                md = joblib.load(meta)
                logger.info(f"Local model loaded from {d}")
                return {"model": model, "meta": md, "dir": str(d)}
            except Exception as e:
                logger.warning(f"Error loading model from {d}: {e}")
                continue
    logger.info("No local model bundle found")
    return None

def call_api_enhanced(api_url: str, payload: dict, timeout: float = 10.0):
    """Enhanced API call with better error handling and timeout"""
    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload, timeout=timeout)
        
        # Check if response is successful
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            result["api_processing_time"] = processing_time
            result["api_status"] = "success"
            logger.info(f"API call successful: {processing_time:.2f}s")
            return result
        else:
            logger.warning(f"API HTTP error: {response.status_code} - {response.text}")
            return {
                "error": f"HTTP {response.status_code}: {response.text}",
                "api_status": "http_error",
                "status_code": response.status_code
            }
        
    except requests.exceptions.Timeout:
        logger.warning(f"API timeout after {timeout}s")
        return {"error": f"API timeout after {timeout}s", "api_status": "timeout"}
    except requests.exceptions.ConnectionError:
        logger.warning("API connection error")
        return {"error": "Connection failed - API server may not be running", "api_status": "connection_error"}
    except requests.exceptions.HTTPError as e:
        logger.warning(f"API HTTP error: {e}")
        return {"error": f"HTTP error: {e}", "api_status": "http_error"}
    except Exception as e:
        logger.warning(f"API call failed: {e}")
        return {"error": str(e), "api_status": "unknown_error"}

def make_enhanced_donut(prob, color="#ef4444"):
    """Enhanced donut chart with better styling"""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Create donut chart
    wedges, _ = ax.pie(
        [prob, 1-prob], 
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        startangle=90,
        colors=[color, '#f3f4f6']
    )
    
    # Add center text
    ax.text(0, 0, f"{prob*100:.0f}%", 
            ha='center', va='center', 
            fontsize=24, weight='bold', 
            color=color)
    
    # Remove axes
    ax.set(aspect="equal")
    plt.tight_layout()
    
    return fig

def get_risk_display_info(probability):
    """Get comprehensive risk display information"""
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

def export_enhanced_csv_bytes(answers: dict, result: dict):
    """Enhanced CSV export with more details"""
    buf = io.StringIO()
    w = csv.writer(buf)
    
    # Header
    w.writerow(["ASD Screening Report", "Generated on", datetime.now().strftime('%Y-%m-%d %H:%M')])
    w.writerow([])
    
    # Answers section
    w.writerow(["Question", "Answer", "Numeric Value"])
    for q, a in answers.items():
        numeric_value = a
        if isinstance(a, int) and q in QUESTION_OPTIONS:
            option_text = QUESTION_OPTIONS[q]["options"].get(a, "Unknown")
            w.writerow([q, option_text, numeric_value])
        else:
            w.writerow([q, str(a), numeric_value])
    
    w.writerow([])
    
    # Results section
    w.writerow(["RESULTS"])
    w.writerow(["Probability", f"{result.get('probability_asd', 0)*100:.1f}%"])
    w.writerow(["Risk Category", result.get('risk_category', 'Unknown')])
    w.writerow(["Risk Description", result.get('risk_description', '')])
    w.writerow(["Explanation", result.get('explanation', '')])
    
    # Data quality if available
    if 'data_quality' in result:
        w.writerow([])
        w.writerow(["DATA QUALITY"])
        w.writerow(["Completion Rate", f"{result['data_quality'].get('completion_percentage', 0):.1f}%"])
        w.writerow(["Risk Indicators", result['data_quality'].get('risk_indicators', 0)])
    
    return buf.getvalue().encode("utf-8")


    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "ASD Screening Report", ln=True, align='C')
    pdf.ln(5)
    
    # Date and info
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M')}", ln=True)
    pdf.cell(0, 8, "Screening Aid Only - Not a Diagnosis", ln=True)
    pdf.ln(10)
    
    # Results section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Screening Results", ln=True)
    pdf.set_font("Arial", size=11)
    
    probability = result.get('probability_asd', 0) * 100
    risk_category = result.get('risk_category', 'Unknown')
    
    pdf.cell(0, 8, f"Probability: {probability:.1f}%", ln=True)
    pdf.cell(0, 8, f"Risk Category: {risk_category}", ln=True)
    pdf.cell(0, 8, f"Description: {result.get('risk_description', '')}", ln=True)
    pdf.ln(5)
    
    # Explanation
    pdf.multi_cell(0, 6, f"Explanation: {result.get('explanation', '')}")
    pdf.ln(10)
    
    # Answers section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Questionnaire Answers", ln=True)
    pdf.set_font("Arial", size=10)
    
    for i, (q, a) in enumerate(answers.items(), 1):
        answer_text = str(a)
        if isinstance(a, int) and q in QUESTION_OPTIONS:
            answer_text = QUESTION_OPTIONS[q]["options"].get(a, "Unknown")
        
        pdf.multi_cell(0, 6, f"{i}. {q}: {answer_text}")
        pdf.ln(1)
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "Important: This screening is not a diagnosis. Please consult with healthcare professionals for comprehensive evaluation.")
    
    return pdf.output(dest='S').encode('latin-1')

# ---------------- FIXED: Simple Progress Tracking ----------------
TOTAL_QUESTIONS = len(QUESTIONS)

def validate_questionnaire_completion(answers):
    """FIXED: Now correctly counts ALL answers including 0 values"""
    total_questions = len(QUESTIONS)
    
    # FIX: Count ALL answers that have been set (including 0)
    answered = sum(1 for a in answers.values() if a is not None)
    completion_rate = (answered / total_questions) * 100
    
    return {
        "completion_rate": completion_rate,
        "total_questions": total_questions,
        "answered_questions": answered,
        "is_adequately_completed": answered == total_questions
    }

# ---------------- Enhanced Session State Management ----------------
def initialize_session_state():
    """Initialize all session state variables with enhanced tracking"""
    default_state = {
        "initialized": True,
        "chat": [("bot", "👋 Welcome! I'm your ASD screening assistant. I'll guide you through 15 simple questions about your child's development. This screening is not a diagnosis but can help identify areas for further evaluation.")],
        "q_index": 0,
        "submitted": False,
        "result": None,
        "answers": {q: None for q in QUESTIONS},  # Start with None for unselected
        "api_status": "unknown",
        "start_time": datetime.now(),
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load local model bundle
local_bundle = load_local_bundle()

# ---------------- Enhanced UI Layout ----------------
st.markdown(f"""
<div class='card'>
    <h1 style='margin-bottom: 0;'>{APP_TITLE}</h1>
    <p class='small-muted'>Enhanced screening tool with AI assistance • Version 2.0</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced information
with st.sidebar:
    st.header("🔧 Settings & Controls")
    
    api_url = st.text_input("Prediction API URL", value=DEFAULT_API, help="URL of the prediction API endpoint")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Health Check", use_container_width=True):
            with st.spinner("Checking API..."):
                try:
                    # Try health endpoint first
                    health_url = api_url.replace("/predict", "/health")
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        st.session_state.api_status = health_data.get("status", "healthy")
                        st.success("✅ API Healthy")
                        if health_data.get("model_loaded"):
                            st.info("🤖 Model: Loaded")
                        else:
                            st.warning("🤖 Model: Not Loaded")
                    else:
                        st.warning(f"⚠️ API Responded with Error: {response.status_code}")
                        st.session_state.api_status = "error"
                except Exception as e:
                    # Try root endpoint as fallback
                    try:
                        root_url = api_url.replace("/predict", "")
                        response = requests.get(root_url, timeout=5)
                        if response.status_code == 200:
                            st.session_state.api_status = "healthy"
                            st.success("✅ API Reachable")
                        else:
                            st.error("❌ API Unreachable")
                            st.session_state.api_status = "unreachable"
                    except:
                        st.error("❌ API Server Not Running")
                        st.session_state.api_status = "unreachable"
                        st.info("💡 Start the API server with: `python agent_api.py`")
    
    with col2:
        if st.button("🗑️ Reset Form", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()
    
    # System information
    st.markdown("---")
    st.subheader("System Information")
    
    if local_bundle:
        st.success(f"✅ Local Model Available")
        st.caption(f"Location: {local_bundle['dir']}")
    else:
        st.info("ℹ️ Using API Mode")
        
    # FIXED: Safe API status check
    api_status = getattr(st.session_state, 'api_status', 'unknown')
    if api_status != "unknown":
        status_color = {
            "healthy": "🟢",
            "degraded": "🟡", 
            "unreachable": "🔴",
            "error": "🔴"
        }.get(api_status, "⚪")
        st.caption(f"API Status: {status_color} {api_status.title()}")
    else:
        st.caption("API Status: ⚪ Unknown")

    # FIXED: Progress in sidebar - Now uses simple answer-based tracking
    st.markdown("---")
    st.subheader("📊 Progress Overview")
    
    completion = validate_questionnaire_completion(st.session_state.answers)
    
    st.metric("Questions Answered", 
              f"{completion['answered_questions']}/{completion['total_questions']}",
              f"{completion['completion_rate']:.0f}%")
    
    st.progress(completion['completion_rate'] / 100)
    
    # Show next action
    if completion['answered_questions'] < TOTAL_QUESTIONS:
        remaining = TOTAL_QUESTIONS - completion['answered_questions']
        st.info(f"🔄 {remaining} question(s) remaining")
    else:
        st.success("✅ All questions answered!")

# Helper chat functions
def append_user(msg: str):
    if not st.session_state.chat or st.session_state.chat[-1] != ("user", msg):
        st.session_state.chat.append(("user", msg))

def append_bot(msg: str):
    if not st.session_state.chat or st.session_state.chat[-1] != ("bot", msg):
        st.session_state.chat.append(("bot", msg))

# Main card — enhanced hybrid UI
st.markdown("<div class='card'>", unsafe_allow_html=True)
left, right = st.columns([2, 1])

with left:
    # Enhanced chat bubbles with typing indicators
    chat_container = st.container()
    with chat_container:
        for who, txt in st.session_state.chat:
            cls = "bot" if who == "bot" else "user"
            st.markdown(f"<div class='chat-bubble {cls} clearfix'>{txt}</div>", unsafe_allow_html=True)
        st.markdown("<div class='clearfix'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # Current question with enhanced display
    idx = st.session_state.q_index
    qtext = QUESTIONS[idx]
    q_config = QUESTION_OPTIONS[qtext]
    
    # FIXED: Simple progress display at top
    st.markdown(f"**Question {idx+1} of {TOTAL_QUESTIONS}**")
    completion = validate_questionnaire_completion(st.session_state.answers)
    st.progress(completion['completion_rate'] / 100)
    
    st.markdown(f"### {qtext}")
    st.caption(f"*{q_config.get('description', 'Behavior assessment')}*")
    
    # Enhanced input with better styling
    options = q_config["options"]
    radio_key = f"radio_{idx}"
    
    # Get current answer (handle None case)
    current_answer = st.session_state.answers[qtext]
    
    # FIXED: Set default index properly
    if current_answer is None:
        default_index = 0  # Select first option by default
    else:
        # Find the index of the current answer in the options keys
        option_keys = list(options.keys())
        default_index = option_keys.index(current_answer) if current_answer in option_keys else 0

    # Show appropriate input based on question type
    choice = st.radio(
        label="Select your response:",
        options=list(options.keys()),
        index=default_index,
        key=radio_key,
        format_func=lambda v: options[v],
        horizontal=False
    )
    
    # FIXED: Update answer immediately when radio changes with proper rerun
    if choice != current_answer:
        st.session_state.answers[qtext] = choice
        # Use callback to force update
        st.rerun()

    # Enhanced navigation
    with st.form(key="nav_form"):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            prev_clicked = st.form_submit_button("⬅ Previous", 
                                               disabled=(idx==0),
                                               use_container_width=True)
        with c2:
            next_clicked = st.form_submit_button("Next ➡", 
                                               disabled=(idx==len(QUESTIONS)-1),
                                               use_container_width=True)
        with c3:
            # Check completion before allowing submit
            completion = validate_questionnaire_completion(st.session_state.answers)
            submit_disabled = not completion['is_adequately_completed']
            
            if submit_disabled:
                submit_label = f"🔒 Complete {completion['answered_questions']}/{completion['total_questions']}"
            else:
                submit_label = "🚀 Submit & Analyze"
                
            submit_clicked = st.form_submit_button(submit_label, 
                                                 disabled=submit_disabled,
                                                 use_container_width=True)
        
        if prev_clicked:
            append_user(f"◀️ Back to question {idx}")
            st.session_state.q_index = max(0, st.session_state.q_index - 1)
            st.rerun()
            
        if next_clicked:
            answer_label = options[choice]
            append_user(f"✅ Answered: {answer_label}")
            st.session_state.q_index = min(len(QUESTIONS)-1, st.session_state.q_index + 1)
            st.rerun()
            
        if submit_clicked:
            append_user("📊 Submitted all answers for analysis")
            st.session_state.submitted = True
            st.rerun()

    # Enhanced answer review
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    with st.expander("📋 Review Your Answers", expanded=False):
        completion = validate_questionnaire_completion(st.session_state.answers)
        
        if completion['completion_rate'] < 100:
            st.warning(f"Questionnaire is {completion['completion_rate']:.0f}% complete. Please answer all questions before submitting.")
        
        for i, (q, a) in enumerate(st.session_state.answers.items(), 1):
            answer_text = "Not answered" if a is None else str(a)
            if isinstance(a, int) and q in QUESTION_OPTIONS:
                answer_text = QUESTION_OPTIONS[q]["options"].get(a, "Unknown")
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.write(f"**Q{i}:** {q}")
            with col2:
                status = "✅" if a is not None else "❌"
                st.write(f"{status} `{answer_text}`")

with right:
    st.markdown("### 📈 Progress")
    
    completion = validate_questionnaire_completion(st.session_state.answers)
    
    # FIXED: Enhanced progress visualization
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0;'>
        <div style='font-size: 2em; font-weight: bold; color: #06b6d4;'>{completion['answered_questions']}</div>
        <div style='color: #6b7280;'>of {completion['total_questions']} questions answered</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(completion['completion_rate'] / 100)
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    st.metric("Completion", f"{completion['completion_rate']:.0f}%")
    st.metric("Questions Answered", f"{completion['answered_questions']}/{TOTAL_QUESTIONS}")
    
    # Time tracking
    if 'start_time' in st.session_state:
        duration = datetime.now() - st.session_state.start_time
        minutes = int(duration.total_seconds() / 60)
        st.metric("Time Elapsed", f"{minutes} min")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Answer based on your child's typical behavior
    - If unsure, choose the closest option
    - All questions must be answered for accurate screening
    - This is a screening tool, not a diagnosis
    """)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Enhanced Submit Handling ----------------
if st.session_state.submitted:
    st.session_state.submitted = False
    
    answers = st.session_state.answers
    completion = validate_questionnaire_completion(st.session_state.answers)
    
    # Enhanced payload with session tracking
    payload = {
        "answers": answers,
        "session_id": f"session_{int(time.time())}",
        "validate_data": True
    }

    # Prediction with enhanced status
    with st.spinner("🔄 Analyzing responses with AI model..."):
        result = call_api_enhanced(api_url, payload, timeout=15.0)

    # Handle API response
    if "error" in result:
        st.error(f"❌ API Error: {result['error']}")
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.write("**Troubleshooting tips:**")
        st.write("1. Check if the API server is running")
        st.write("2. Verify the API URL in settings")
        st.write("3. Try the local model fallback below")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced fallback to local model
        st.warning("Attempting local model fallback...")
        if local_bundle:
            try:
                with st.spinner("🔄 Using local model..."):
                    model = local_bundle["model"]
                    meta = local_bundle["meta"]
                    feat_cols = meta.get("feature_cols", QUESTIONS)
                    
                    # Convert answers to feature vector
                    vec = []
                    for c in feat_cols:
                        v = answers.get(c, 0)  # Use 0 if None
                        try:
                            vec.append(float(v))
                        except:
                            vec.append(0.0)
                    
                    X = np.array(vec).reshape(1, -1)
                    
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X)[0, 1])
                    else:
                        prob = float(model.predict(X)[0])
                    
                    # Enhanced risk assessment
                    cat, color, description = get_risk_display_info(prob)
                    
                    result = {
                        "probability_asd": prob,
                        "risk_category": cat,
                        "risk_color": color,
                        "risk_description": description,
                        "explanation": f"Local model analysis indicates {cat.lower()} risk level.",
                        "features_used": {c: answers.get(c, None) for c in feat_cols},
                        "data_quality": completion,
                        "prediction_metadata": {"method": "local_fallback"},
                        "model_info": {"model_type": type(model).__name__}
                    }
                    
                    st.success("✅ Local prediction completed")
                    
            except Exception as e:
                st.error(f"❌ Local model error: {e}")
                result = None
        else:
            st.error("No local model available for fallback.")
            result = None
    else:
        # API call was successful
        st.success("✅ Analysis completed successfully!")

    # Enhanced result display
    if result and "probability_asd" in result:
        st.session_state.result = result
        prob = float(result.get("probability_asd", 0.0))
        
        # Get risk information
        if "risk_color" in result:
            cat = result.get("risk_category", "Unknown")
            color = result.get("risk_color", "#6b7280")
            description = result.get("risk_description", "")
        else:
            cat, color, description = get_risk_display_info(prob)
        
        explanation = result.get("explanation", "Analysis completed.")
        
        # Enhanced chat update
        append_bot(f"**Analysis Complete:** {cat} risk level detected ({prob*100:.1f}% probability)")
        
        # Enhanced result display
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## 📊 Screening Results")
        
        # Result header
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Risk Assessment")
            st.pyplot(make_enhanced_donut(prob, color))
            st.markdown(f"<div class='risk-pill' style='background:{color}'>{cat}</div>", unsafe_allow_html=True)
            st.metric("Probability", f"{prob*100:.1f}%")
            st.caption(description)
        
        with col2:
            st.markdown("### 📝 Detailed Analysis")
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.write(explanation)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Next steps based on risk level
            st.markdown("#### 🎯 Recommended Next Steps")
            if "Low" in cat:
                st.success("""
                - Continue with regular developmental monitoring
                - Engage in play-based social interactions
                - Schedule routine pediatric check-ups
                - Monitor language and social milestones
                """)
            elif "Moderate" in cat:
                st.warning("""
                - Consider early intervention consultation
                - Increase social interaction opportunities
                - Document specific concerns for professional evaluation
                - Schedule developmental screening with pediatrician
                """)
            else:
                st.error("""
                - Seek comprehensive evaluation from developmental specialist
                - Contact early intervention services
                - Consider speech and language assessment
                - Discuss concerns with healthcare provider promptly
                """)
        
        # Enhanced data quality information
        if 'data_quality' in result:
            st.markdown("---")
            st.markdown("### 📈 Data Quality")
            dq = result['data_quality']
            if isinstance(dq, dict):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Completion", f"{dq.get('completion_percentage', 0):.0f}%")
                with col2:
                    st.metric("Risk Indicators", dq.get('risk_indicators', 0))
                with col3:
                    st.metric("Questions", f"{dq.get('answered_questions', 0)}/{dq.get('total_questions', 15)}")
        
        # Features used (collapsible)
        with st.expander("🔍 View Technical Details"):
            st.json(result.get("features_used", {}))
            
            if 'prediction_metadata' in result:
                st.write("**Prediction Metadata:**")
                st.write(result['prediction_metadata'])
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced export options
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_bytes = export_enhanced_csv_bytes(answers, result)
            st.download_button(
                "📥 Download CSV Report", 
                data=csv_bytes, 
                file_name=f"asd_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                mime="text/csv",
                use_container_width=True
            )
        

# Enhanced Footer
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>Important Disclaimer:</strong> This screening tool is for informational purposes only and is not a diagnostic tool. 
    It does not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for any health concerns or before making any medical decisions.</p>
    <p>ASD Early Screening Assistant • Version 2.0 • Enhanced AI Model</p>
</div>
""", unsafe_allow_html=True)