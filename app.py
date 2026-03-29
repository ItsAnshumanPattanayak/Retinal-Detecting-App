#!/usr/bin/env python3
"""
MediAgent-Retina: AI-Powered Retinal Disease Detection
Frontend Application - Streamlit Interface
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Install missing packages
def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install_if_missing("streamlit")
install_if_missing("timm")
install_if_missing("albumentations")

import streamlit as st
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import agents (assuming they're in a separate file)
try:
    from agents import (
        ReportAgent, PrecautionsAgent, DiseaseDetailsAgent,
        DemographicsAgent, ChatbotAgent, DISEASE_KNOWLEDGE_BASE
    )
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    # Fallback: Define minimal knowledge base
    DISEASE_KNOWLEDGE_BASE = {
        "Normal": {"description": "Healthy retina", "severity": "None", "urgency": "Routine", "color": "#4CAF50"},
        "Diabetic Retinopathy": {"description": "Diabetes complication", "severity": "High", "urgency": "URGENT", "color": "#f44336"},
        "ARMD": {"description": "Age-related macular degeneration", "severity": "Moderate", "urgency": "HIGH", "color": "#FF5722"},
        "Media Haze": {"description": "Ocular media opacity", "severity": "Mild", "urgency": "Moderate", "color": "#FF9800"},
        "Optic Disc Disease": {"description": "Optic nerve abnormality", "severity": "Moderate", "urgency": "HIGH", "color": "#E91E63"},
        "Retinal Vascular Disease": {"description": "Retinal blood vessel issues", "severity": "Severe", "urgency": "URGENT", "color": "#9C27B0"},
        "Myopia": {"description": "Pathological myopia", "severity": "Mild", "urgency": "Moderate", "color": "#2196F3"},
        "Other Diseases": {"description": "Other retinal pathology", "severity": "Variable", "urgency": "Moderate", "color": "#607D8B"}
    }

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

class CFG:
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "Normal", "Diabetic Retinopathy", "ARMD", "Media Haze",
        "Optic Disc Disease", "Retinal Vascular Disease", "Myopia", "Other Diseases"
    ]
    IMAGE_SIZE = 224
    CHECKPOINT_PATH = "./checkpoints/ensemble_best.pth"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediAgent-Retina | AI Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .agent-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=False, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, 
                                          num_classes=0, drop_rate=dropout)
        self.feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim), nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 512), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, 
                                          num_classes=0, drop_rate=dropout)
        self.feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feat_dim), nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 512), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('vgg16', pretrained=pretrained, 
                                          num_classes=0, drop_rate=dropout)
        self.feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim), nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 512), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

class RetinalEnsemble(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super().__init__()
        self.efficientnet = EfficientNetClassifier(num_classes, pretrained, 0.4)
        self.resnet = ResNetClassifier(num_classes, pretrained, 0.5)
        self.vgg = VGGClassifier(num_classes, pretrained, 0.5)
        self.weights = nn.Parameter(torch.tensor([0.45, 0.35, 0.20]))

    def forward(self, x, return_individual=False):
        logits_eff = self.efficientnet(x)
        logits_res = self.resnet(x)
        logits_vgg = self.vgg(x)
        probs_eff = F.softmax(logits_eff, dim=1)
        probs_res = F.softmax(logits_res, dim=1)
        probs_vgg = F.softmax(logits_vgg, dim=1)
        w = F.softmax(self.weights, dim=0)
        ensemble_probs = w[0] * probs_eff + w[1] * probs_res + w[2] * probs_vgg
        if return_individual:
            return ensemble_probs, {
                'efficientnet': {'probs': probs_eff, 'logits': logits_eff},
                'resnet': {'probs': probs_res, 'logits': logits_res},
                'vgg': {'probs': probs_vgg, 'logits': logits_vgg}
            }
        return ensemble_probs

@st.cache_resource
def load_model():
    model = RetinalEnsemble(CFG.NUM_CLASSES, pretrained=False)
    model_loaded = False
    if os.path.exists(CFG.CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CFG.CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_loaded = True
        except:
            pass
    model = model.to(DEVICE)
    model.eval()
    return model, model_loaded

# ════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ════════════════════════════════════════════════════════════════════

def preprocess_fundus(image, size=224):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w > image.shape[1] * 0.3 and h > image.shape[0] * 0.3:
            image = image[y:y+h, x:x+w]
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(image, (0, 0), 10)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

def get_transforms(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])

def check_image_quality(image):
    issues = []
    score = 100
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        issues.append(f"Image too small ({w}×{h})")
        score -= 30
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 30:
        issues.append("Too dark")
        score -= 20
    elif mean_brightness > 230:
        issues.append("Too bright")
        score -= 20
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        issues.append("Blurry")
        score -= 25
    return {
        'passed': max(0, score) >= 50,
        'score': max(0, score),
        'resolution': f'{w}×{h}',
        'brightness': mean_brightness,
        'sharpness': laplacian_var,
        'issues': issues if issues else ['Good quality']
    }

def predict(model, image):
    processed = preprocess_fundus(image, CFG.IMAGE_SIZE)
    display_img = processed.astype(np.float32) / 255.0
    transform = get_transforms(CFG.IMAGE_SIZE)
    tensor = transform(image=processed)['image'].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        ensemble_probs, individual = model(tensor, return_individual=True)

    probs = ensemble_probs[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    individual_results = {}
    for name, data in individual.items():
        p = data['probs'][0].cpu().numpy()
        idx = int(p.argmax())
        individual_results[name] = {
            'prediction': CFG.CLASS_NAMES[idx],
            'confidence': float(p[idx]),
            'probabilities': p,
            'pred_idx': idx
        }

    preds = [individual_results[m]['prediction'] for m in ['efficientnet', 'resnet', 'vgg']]
    unanimous = len(set(preds)) == 1

    return {
        'predicted_class': CFG.CLASS_NAMES[pred_idx],
        'pred_idx': pred_idx,
        'confidence': confidence,
        'probabilities': probs,
        'individual': individual_results,
        'display_image': display_img,
        'tensor': tensor,
        'unanimous': unanimous
    }

# ════════════════════════════════════════════════════════════════════
# INITIALIZE AGENTS
# ════════════════════════════════════════════════════════════════════

if AGENTS_AVAILABLE:
    report_agent = ReportAgent()
    precautions_agent = PrecautionsAgent()
    disease_agent = DiseaseDetailsAgent()
    demographics_agent = DemographicsAgent()
    chatbot_agent = ChatbotAgent()

# ════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ MediAgent-Retina</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Retinal Disease Detection System</p>', 
                unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════

    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown(f"**Device:** `{DEVICE}`")

    with st.spinner("Loading AI models..."):
        model, model_loaded = load_model()

    if model_loaded:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.warning("⚠️ Running in demo mode")

    st.sidebar.markdown("---")

    if AGENTS_AVAILABLE:
        st.sidebar.subheader("🤖 AI Agents Active")
        st.sidebar.markdown("""
        - 📋 Report Generator
        - ⚠️ Precautions Advisor
        - 🔬 Disease Information
        - 👥 Demographics Analyst
        - 💬 Interactive Chatbot
        """)
    else:
        st.sidebar.warning("⚠️ Agents module not found")

    st.sidebar.markdown("---")
    patient_id = st.sidebar.text_input("Patient ID (optional)", placeholder="P12345")

    # ═══════════════════════════════════════════════════════════════
    # MAIN CONTENT - UPLOAD & ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Fundus Image")

        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear fundus photograph"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Convert to RGB
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Quality check
            quality = check_image_quality(image_np)

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.metric("Quality Score", f"{quality['score']}/100")
            with col_q2:
                st.metric("Resolution", quality['resolution'])

            if quality['passed']:
                st.success("✅ Image quality: Good")
            else:
                st.warning(f"⚠️ Quality issues: {', '.join(quality['issues'])}")

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing retinal image..."):
                    result = predict(model, image_np)
                    st.session_state['result'] = result
                    st.session_state['quality'] = quality
                    st.success("✅ Analysis complete!")

    with col2:
        st.subheader("📋 Diagnosis Results")

        if 'result' in st.session_state:
            result = st.session_state['result']
            disease = result['predicted_class']
            confidence = result['confidence']
            info = DISEASE_KNOWLEDGE_BASE.get(disease, {})

            # Main diagnosis box
            st.markdown(f"""
            <div class="diagnosis-box">
                <h2 style="margin:0; font-size:2rem">{disease}</h2>
                <p style="font-size:1.5rem; margin:10px 0">
                    Confidence: <strong>{confidence:.1%}</strong>
                </p>
                <p style="margin:5px 0">
                    Severity: <strong>{info.get('severity', 'N/A')}</strong> | 
                    Urgency: <strong>{info.get('urgency', 'N/A')}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Model agreement
            st.markdown("### 🤖 Model Consensus")
            if result['unanimous']:
                st.success("✅ All 3 models agree on this diagnosis")
            else:
                st.warning("⚠️ Models show partial disagreement")

            # Individual model predictions
            with st.expander("📊 View Individual Model Predictions"):
                for name, res in result['individual'].items():
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.write(f"**{name.title()}:** {res['prediction']}")
                    with col_b:
                        st.write(f"{res['confidence']:.1%}")

            # Probability distribution
            with st.expander("📈 View Probability Distribution"):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(CFG.CLASS_NAMES, result['probabilities'], color='#667eea')
                ax.set_xlabel('Probability')
                ax.set_title('Class Probabilities')
                st.pyplot(fig)

        else:
            st.info("👆 Upload and analyze an image to see results")

    # ═══════════════════════════════════════════════════════════════
    # AI AGENTS SECTION
    # ═══════════════════════════════════════════════════════════════

    if 'result' in st.session_state and AGENTS_AVAILABLE:
        st.markdown("---")
        st.markdown("## 🤖 AI Agents Analysis")

        result = st.session_state['result']
        quality = st.session_state['quality']
        disease = result['predicted_class']

        tabs = st.tabs([
            "📋 Report",
            "⚠️ Precautions", 
            "🔬 Disease Info",
            "👥 Demographics",
            "💬 Chatbot"
        ])

        # ═══════════════════════════════════════════════════════════
        # Agent 1: Report
        # ═══════════════════════════════════════════════════════════
        with tabs[0]:
            st.markdown('<div class="agent-box"><h3>📋 Medical Analysis Report</h3></div>', 
                       unsafe_allow_html=True)

            report = report_agent.generate(result, quality, patient_id)
            st.markdown(report)

            st.download_button(
                "📥 Download Report (Markdown)",
                report,
                f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown",
                use_container_width=True
            )

        # ═══════════════════════════════════════════════════════════
        # Agent 2: Precautions
        # ═══════════════════════════════════════════════════════════
        with tabs[1]:
            st.markdown('<div class="agent-box"><h3>⚠️ Safety Instructions</h3></div>', 
                       unsafe_allow_html=True)

            precautions = precautions_agent.generate(disease)
            st.markdown(precautions)

        # ═══════════════════════════════════════════════════════════
        # Agent 3: Disease Details
        # ═══════════════════════════════════════════════════════════
        with tabs[2]:
            st.markdown('<div class="agent-box"><h3>🔬 Comprehensive Disease Information</h3></div>', 
                       unsafe_allow_html=True)

            disease_info = disease_agent.generate(disease)
            st.markdown(disease_info)

        # ═══════════════════════════════════════════════════════════
        # Agent 4: Demographics
        # ═══════════════════════════════════════════════════════════
        with tabs[3]:
            st.markdown('<div class="agent-box"><h3>👥 Age & Demographics</h3></div>', 
                       unsafe_allow_html=True)

            demographics = demographics_agent.generate(disease)
            st.markdown(demographics)

        # ═══════════════════════════════════════════════════════════
        # Agent 5: Chatbot
        # ═══════════════════════════════════════════════════════════

        with tabs[4]:
            # ✅ ADD THIS CSS FIX RIGHT HERE AT THE TOP
            st.markdown("""
            <style>
                .chat-message {
                    color: #000000 !important;
                }
                .chat-message * {
                    color: #000000 !important;
                }
                .user-message {
                    background-color: #EBF5FF !important;
                    color: #000000 !important;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                }
                .bot-message {
                    background-color: #F0F0F0 !important;
                    color: #000000 !important;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                }
                .user-message strong,
                .bot-message strong {
                    color: #000000 !important;
                }
        </style>
        """, unsafe_allow_html=True)







    # ... rest of your code stays the same
            st.markdown('<div class="agent-box"><h3>💬 Ask MediBot</h3></div>', 
                       unsafe_allow_html=True)

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong> {msg["content"]}</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message bot-message"><strong>🤖 MediBot:</strong> {msg["content"]}</div>', 
                                   unsafe_allow_html=True)

            # Chat input
            col_chat1, col_chat2 = st.columns([4, 1])
            with col_chat1:
                user_input = st.text_input("Your question:", key="chat_input", label_visibility="collapsed", 
                                          placeholder="Ask about symptoms, treatment, prevention...")
            with col_chat2:
                send_btn = st.button("Send 📤", use_container_width=True)

            if st.button("Clear Chat 🗑️"):
                st.session_state.chat_history = []
                st.rerun()

            if send_btn and user_input:
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                response = chatbot_agent.get_response(user_input, disease)
                st.session_state.chat_history.append({'role': 'bot', 'content': response})
                st.rerun()

            # Quick questions
            st.markdown("### 💡 Quick Questions")
            quick_qs = [
                f"What are symptoms of {disease}?",
                f"How to treat {disease}?",
                "What foods are good for eye health?",
                "How to prevent eye disease?",
                "When should I see a doctor?"
            ]

            cols = st.columns(3)
            for i, q in enumerate(quick_qs):
                with cols[i % 3]:
                    if st.button(q[:35] + "..." if len(q) > 35 else q, key=f"q_{i}", use_container_width=True):
                        st.session_state.chat_history.append({'role': 'user', 'content': q})
                        response = chatbot_agent.get_response(q, disease)
                        st.session_state.chat_history.append({'role': 'bot', 'content': response})
                        st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
        This AI system is for <strong>screening purposes only</strong> and does NOT replace 
        professional medical diagnosis. Always consult a qualified ophthalmologist for medical advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; color:#888; margin-top:30px; font-size:0.9rem'>
        🏥 MediAgent-Retina v2.0 | 3-Model Ensemble + 5 AI Agents<br>
        EfficientNetB3 + ResNet50 + VGG16 | Built with PyTorch & Streamlit
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
