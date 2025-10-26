import streamlit as st
import joblib
import numpy as np
import os

# ---------------- page config ----------------
st.set_page_config(page_title="Parkinson Detection", layout="wide", initial_sidebar_state="expanded")

# ----------------- style (dark tech look) -----------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        
        background-size: cover;
        background-position: center;
        color: #e6f7ff;
        font-family: 'Poppins', sans-serif;
    }

    /* Card box */
    .card {
        background: rgba(6,10,13,0.7);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        border: 1px solid rgba(0,224,255,0.12);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00e0ff, #00a3d9);
        color: #021018;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 10px 16px;
    }
    div.stButton > button:hover {
        filter: brightness(0.95);
    }

    /* Result boxes */
    .result-success {
        background: linear-gradient(90deg, rgba(0,224,255,0.08), rgba(0,160,217,0.06));
        border-left: 4px solid #00e0ff;
        padding: 12px;
        border-radius: 8px;
    }
    .result-warn {
        background: linear-gradient(90deg, rgba(255,200,50,0.06), rgba(255,160,10,0.04));
        border-left: 4px solid #ffc107;
        padding: 12px;
        border-radius: 8px;
    }
    .result-error {
        background: linear-gradient(90deg, rgba(255,80,80,0.06), rgba(255,40,40,0.04));
        border-left: 4px solid #ff4d4d;
        padding: 12px;
        border-radius: 8px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #00e0ff;
        text-align: center;
    }

    /* Small responsive fix */
    @media (max-width: 800px) {
        .card { padding: 15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- load model safely -----------------
MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\data analyst\parkinson_.pkl"
SCALER_PATH = r"C:\Users\asus\OneDrive\Desktop\data analyst\scaler.pkl"

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        st.sidebar.error(f"Model not found at: {MODEL_PATH}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        st.sidebar.error(f"Scaler not found at: {SCALER_PATH}")
except Exception as e:
    st.sidebar.error(f"Error loading scaler: {e}")

# ----------------- sidebar navigation -----------------

st.sidebar.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #010b12, #021c28);
        color: #e6f7ff;
        padding: 20px 10px;
    }

    /* Title style */
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #00e0ff;
        text-align: center;
        margin-bottom: 15px;
    }

    /* Radio button labels */
    div[role="radiogroup"] label {
        font-size: 18px !important;
        color: #cce7ff !important;
        font-weight: 500;
        padding: 8px 10px;
        border-radius: 8px;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(0, 224, 255, 0.1);
        color: #00e0ff !important;
        transition: 0.3s;
    }

    /* Selected tab */
    div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: rgba(0, 224, 255, 0.15);
        color: #00e0ff !important;
        font-weight: 700;
    }

    /* Divider line */
    hr {
        border: 1px solid rgba(0, 224, 255, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Navigation slider")
page = st.sidebar.radio("", ("Home", "About", "Test"))

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ This tool is for **screening only** — not a diagnosis.")


# ----------------- Home page -----------------
if page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h1>🧠 Parkinson’s Disease Detection Model</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="text-align:center; font-size:20px;">
        This AI-powered model helps analyze speech-derived medical parameters to identify early patterns of Parkinson’s disease.  
        <br><br>
        Use the <b>Test</b> section to enter the voice-derived values and see predictions instantly.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- About page -----------------
elif page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("About Parkinson's Disease 🧠")
    st.markdown(
        """
        **What is Parkinson’s Disease?**  
Parkinson’s Disease (PD) is a **progressive neurological disorder** that affects the brain’s ability to control movement.  
It occurs when nerve cells in a part of the brain called the *substantia nigra* become damaged or die, causing a **reduction in dopamine**, the chemical messenger responsible for smooth and coordinated muscle motion.  

As dopamine levels decrease, communication between brain regions becomes impaired — resulting in symptoms such as:  
- **Tremor:** Uncontrollable shaking, often in the hands or fingers.  
- **Bradykinesia:** Slowness of movement, making everyday actions take longer.  
- **Rigidity:** Muscle stiffness or tightness that limits motion.  
- **Postural instability:** Problems with balance and coordination.  
- **Changes in voice and facial expressions:** Speech may become softer, slower, or more monotone.  

While there is **no permanent cure** for Parkinson’s, early detection allows for timely medical management, lifestyle changes, and physical therapy — all of which can significantly improve quality of life.

---

**How this App Helps 💻**  
This application is designed as a **machine learning-based screening tool** to identify early signs of Parkinson’s Disease using **voice-derived biomarkers**.  
Researchers have discovered that people with Parkinson’s often show subtle vocal changes even before visible motor symptoms appear.  

- The ML model analyzes parameters such as **jitter**, **pitch variation**, **harmonic ratio**, and **PPE (Pitch Period Entropy)** — which can indicate irregularities in voice patterns.  
- Based on these values, the app predicts whether Parkinson’s is **likely detected** or **not detected**.  
- This tool is meant for **educational and preliminary screening purposes only** — it does **not replace professional medical diagnosis**.

---

**Data & Feature Extraction 🧩**  
In practical research, voice data is recorded using a microphone or speech dataset. Specialized tools such as **Praat**, **Parselmouth**, or **Librosa** are then used to extract numerical features like:  
- Fundamental frequency (Fo)  
- Amplitude and jitter measures  
- Formants and harmonics-to-noise ratio (HNR)  
- Entropy and variability metrics  

These extracted features are standardized (scaled) and fed into the machine learning model — typically trained using algorithms like **SVM (Support Vector Machine)** or **Random Forest Classifier**.

---

**Future Improvements 🚀**  
This app can be extended to include:  
- 🎙️ **Audio Upload Feature:** Allowing users to upload `.wav` files to automatically extract features and get predictions.  
- 📊 **Visual Analytics:** Graphical insights showing how each feature contributes to detection.  
- 🧠 **Deep Learning Integration:** Using neural networks (e.g., CNN or LSTM) to analyze voice spectrograms directly.  
- 📱 **Responsive Web App:** A clean, mobile-friendly interface for broader accessibility.  

---

**Disclaimer ⚠️**  
This tool is intended **only for research, learning, and awareness purposes**. It should **not be used for clinical diagnosis**.  
Always consult a certified neurologist for any medical concerns related to Parkinson’s Disease.

        """
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# ----------------- Test page (main prediction form) -----------------
elif page == "Test":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🧪 Parkinson’s Disease Test Section")
    st.write("Enter your voice-derived parameter values below and click **Predict**.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            PPE = st.number_input("PPE", min_value=0.0, format="%.6f", value=0.000000)
            MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.6f", value=0.000000)
            spread1 = st.number_input("spread1", min_value=0.0, format="%.6f", value=0.000000)
        with col2:
            MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.6f", value=0.000000)
            Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0, format="%.6f", value=0.000000)
            MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.6f", value=0.000000)
        with col3:
            spread2 = st.number_input("spread2", min_value=0.0, format="%.6f", value=0.000000)
            st.markdown("###")


        submit = st.form_submit_button("Predict")
        st.markdown("**Tip:** If you don't have real values, use sample data for testing.")


    if submit:
        if model is None or scaler is None:
            st.warning("Model or scaler not loaded — prediction can't run. Check paths in the script.")
        else:
            try:
                input_data = np.array([[PPE, MDVP_Fo, spread1, MDVP_Flo, Jitter_DDP, MDVP_Fhi, spread2]])
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)

                if int(prediction[0]) == 1:
                    st.markdown('<div class="result-error">', unsafe_allow_html=True)
                    st.error("⚠️ Parkinson’s Disease Detected — please consult a neurologist.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success">', unsafe_allow_html=True)
                    st.success("✅ No Parkinson’s Detected (screening).")
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------- footer ---------------
st.markdown(
    """
    <div style="position:fixed;right:18px;bottom:12px;color:#9fdcff;">
        <small>Developed by Sarthak • Demo tool — not a medical device</small>
    </div>
    """,
    unsafe_allow_html=True,
)
