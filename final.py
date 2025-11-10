import streamlit as st
import pandas as pd
import string
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # New import for interactive charts

# ML & DL Imports (Kept for functionality, no changes needed)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️", layout="wide")

# -------------------- ADVANCED NEON DARK THEME STYLING --------------------
st.markdown("""
<style>
/* --- 1. GENERAL BACKGROUND & TEXT --- */
.stApp {
    /* Deeper, richer dark background with subtle blue-violet gradient */
    background: linear-gradient(135deg, #0a0e14 0%, #171d2b 50%, #202c3e 100%);
    color: #E0E0E0; /* Light grey text */
    font-family: 'Inter', sans-serif;
}

/* --- 2. SIDEBAR --- */
section[data-testid="stSidebar"] {
    background: #0d1117; /* GitHub Dark Color */
    color: #F0F0F0;
    border-right: 2px solid #00FFFF; /* Bright cyan border */
    box-shadow: 2px 0px 15px rgba(0, 255, 255, 0.15); /* Soft neon glow */
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
    color: #00FFC2 !important; /* Neon mint for headers */
    text-shadow: 0 0 5px rgba(0, 255, 194, 0.5);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #00FFFF !important; /* Cyan for radio buttons */
    font-weight: 600;
}

/* --- 3. HEADERS & TITLES --- */
h1, h2, h3, h4 {
    color: #00FFFF; /* Bright Cyan Neon */
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(0, 255, 255, 0.7); /* Neon glow effect */
}
/* Sub-header on detector page */
.stMarkdown h2 { 
    color: #FF66CC; /* Neon Pink */
    text-shadow: 0 0 6px rgba(255, 102, 204, 0.6);
}

/* --- 4. CARDS (Container for main content) --- */
.card {
    /* Semi-transparent dark card with a bright border */
    background-color: rgba(18, 25, 41, 0.95);
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0px 0px 25px rgba(0, 255, 255, 0.3); /* Stronger overall glow */
    border: 1px solid rgba(0, 255, 255, 0.5);
    transition: all 0.3s ease-in-out;
}
.card:hover {
    box-shadow: 0px 0px 35px rgba(0, 255, 255, 0.5); /* Subtle hover glow */
}

/* --- 5. BUTTONS (Call to Action) --- */
.stButton>button {
    background: linear-gradient(to right, #00FFFF, #00AFFF); /* Cyan to Sky Blue */
    color: #0d1117; /* Dark text on bright button */
    font-weight: 700;
    border-radius: 15px; /* More rounded */
    border: none;
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0px 5px 20px rgba(0, 255, 255, 0.5); /* Strong box shadow */
    padding: 0.5rem 1.5rem;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02); /* Slight lift and scale */
    box-shadow: 0px 8px 30px rgba(0, 255, 255, 0.9); /* Intense glow on hover */
    background: linear-gradient(to right, #00AFFF, #00FFFF); /* Reverse gradient on hover */
}

/* --- 6. TEXT INPUT / TEXTAREA --- */
textarea, .stSelectbox {
    border-radius: 12px !important;
    border: 2px solid #FF66CC !important; /* Neon Pink border */
    background-color: #1a222f !important; /* Darker input background */
    color: #F0F0F0 !important;
    box-shadow: 0 0 10px rgba(255, 102, 204, 0.4);
}
.stSelectbox div[role="button"] {
    background-color: #1a222f !important;
    color: #F0F0F0 !important;
}

/* --- 7. UTILITY & FEEDBACK --- */
/* Progress bar - Advanced animation look */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(to right, #FF66CC, #00FFFF, #00AFFF); /* Tri-color glow */
    animation: glow-move 2s infinite alternate;
}
/* Keyframe animation for progress bar */
@keyframes glow-move {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

/* Divider */
hr {
    border: 1px solid #FF66CC; /* Neon Pink Divider */
    margin: 20px 0;
}

/* Success/Warning Messages */
.stAlert, .stInfo {
    border-radius: 10px;
    font-weight: 600;
}
.stWarning {
    background-color: rgba(255, 153, 0, 0.1);
    border-left: 5px solid #FF9900;
    color: #FF9900;
}
.stSuccess {
    background-color: rgba(0, 255, 136, 0.1);
    border-left: 5px solid #00FF88;
    color: #00FF88;
}

/* Expander Header */
.streamlit-expanderHeader {
    background-color: #171d2b !important;
    color: #00FFFF !important;
    border-radius: 10px;
    padding: 15px;
    font-weight: 600;
    border: 1px solid rgba(0, 255, 255, 0.3);
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align:center; font-size:60px; text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF;'>📰 FAKE NEWS DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#FF66CC;'>AI-powered detection: <span style='color:#00ff88; font-weight:bold;'>REAL</span> vs <span style='color:#ff4d4d; font-weight:bold;'>FAKE</span> analysis</h4>", unsafe_allow_html=True)
st.write("---")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209265.png", width=120)
    st.markdown("## ⚙️ App Controls")
    
    # Use index=1 to land directly on the detector page for a more interactive first impression
    menu = st.radio("", ["🏠 Home", "🧠 Detector", "📊 Model Info", "ℹ️ About"], index=1) 
    
    st.write("---")
    st.markdown("### 👩‍💻 Developers")
    st.markdown("Lakshya Pratap Singh")
    st.markdown("📧 <span style='color:#FF66CC;'>Lakshya12112@gmail.com</span>", unsafe_allow_html=True)
    st.write("---")
    st.info("💡 **Pro Tip:** Paste short headlines/summaries for quick and accurate results!")

# -------------------- LOAD DATA --------------------
# @st.cache_resource is the recommended replacement for @st.cache
@st.cache_resource
def load_data():
    # Placeholder for file path - IMPORTANT: The user needs to change this path to a valid one!
    try:
        df = pd.read_csv("real_fake news.csv") # Assumes the file is in the same directory as the script
    except FileNotFoundError:
        st.error("🚨 Dataset not found! Please update the file path in the code.")
        df = pd.DataFrame({'Text': ['Sample real news', 'Sample fake news'], 'label': ['Real', 'Fake']})

    df['clean_text'] = df['Text'].apply(lambda x: str(x).lower().translate(str.maketrans('', '', string.punctuation)))
    return df

dataset = load_data()

# -------------------- TRAIN CLASSICAL MODELS --------------------
@st.cache_resource
def train_classical_models(df):
    if len(df) < 2: return None, None, None, 0.0, 0.0, "N/A", "N/A" # Handle minimal data case

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1500, stop_words='english')
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label'].map({'Fake':0, 'Real':1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    rf_report = classification_report(y_test, rf_model.predict(X_test), output_dict=True)

    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    lr_report = classification_report(y_test, lr_model.predict(X_test), output_dict=True)

    return rf_model, lr_model, tfidf, rf_acc, lr_acc, rf_report, lr_report

rf_model, lr_model, vectorizer, rf_acc, lr_acc, rf_report, lr_report = train_classical_models(dataset)

# -------------------- TRAIN CNN MODEL --------------------
@st.cache_resource
def train_cnn(df, max_words=5000, max_len=200):
    if len(df) < 2: return None, None, 200, 0.0 # Handle minimal data case
    
    y = df['label'].map({'Fake':0, 'Real':1}).values
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_text'])
    X_seq = tokenizer.texts_to_sequences(df['clean_text'])
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Only train if data is sufficient
    if len(X_train) > 0:
        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
        cnn_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    else:
        cnn_acc = 0.0

    return model, tokenizer, max_len, cnn_acc

cnn_model, cnn_tokenizer, cnn_max_len, cnn_acc = train_cnn(dataset)

# -------------------- MAIN CONTENT PAGES --------------------
if menu == "🏠 Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🌟 Welcome to the Detector Hub!", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1.5])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3311/3311573.png", width=300)
        
    with col2:
        st.markdown("""
        ### 🔍 What is Fake News?
        <span style='color:#E0E0E0;'>Fake news refers to **false or misleading information** intended to deceive readers. 
        It spreads rapidly and can severely manipulate public opinion.</span>
        
        ### 💡 App Capabilities:
        * **Deep Learning:** Utilizes a **1D CNN** for advanced textual feature extraction.
        * **Machine Learning:** Supports **Random Forest** and **Logistic Regression** (TF-IDF based).
        * **Interactivity:** Real-time analysis with visual feedback.
        """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("<h3 style='color:#00FFC2;'>📊 Data Insights</h3>", unsafe_allow_html=True)

    with st.expander("📄 View Sample Dataset & Stats"):
        st.dataframe(dataset.head(), use_container_width=True)

        # Plotly for interactive chart
        counts = dataset['label'].value_counts().reset_index()
        counts.columns = ['Label', 'Count']
        
        fig = px.bar(counts, 
                     x='Label', 
                     y='Count', 
                     color='Label',
                     color_discrete_map={'Real': '#00ff88', 'Fake': '#ff4d4d'},
                     title="Distribution of Real vs Fake News Articles",
                     labels={'Label': 'News Category', 'Count': 'Number of Articles'})
        
        # Plotly Theme/Style Customization for Dark Mode
        fig.update_layout(
            plot_bgcolor='#1a222f', 
            paper_bgcolor='#1a222f',
            font_color='#E0E0E0',
            title_font_color="#00FFFF",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend_title_text='Category'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🧠 Detector":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>✍️ Paste News for Real-Time Analysis</h2>", unsafe_allow_html=True)
    
    input_text = st.text_area(" ", placeholder="Type or paste your news article here (e.g., a headline or a short summary)...", height=200)

    colA, colB = st.columns([1, 2])
    with colA:
        model_choice = st.selectbox("Select Prediction Model:", ["CNN", "Random Forest", "Logistic Regression"], index=0)
    with colB:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🚀 INITIATE SCAN (ANALYZE NEWS)", use_container_width=True):
            if input_text.strip() == "":
                st.warning("⚠️ **Input Required!** Please enter some text before analysis.")
            else:
                # --- Interactive Loading Simulation ---
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                    if i < 30:
                        progress_text.text(f"🔍 Initializing Text Preprocessing... {i}%")
                    elif i < 70:
                        progress_text.text(f"🧠 Running {model_choice} Model Prediction... {i}%")
                    else:
                        progress_text.text(f"✅ Finalizing Report Generation... {i}%")
                
                progress_text.empty()
                progress_bar.empty()
                
                # --- Prediction Logic (Unchanged) ---
                clean_input = input_text.lower().translate(str.maketrans('', '', string.punctuation))

                pred_status = "UNKNOWN"
                acc = 0.0

                if model_choice == "Random Forest":
                    if vectorizer and rf_model:
                        input_vec = vectorizer.transform([clean_input])
                        pred = rf_model.predict(input_vec)[0]
                        acc = rf_acc
                        pred_status = "REAL" if pred == 1 else "FAKE"
                
                elif model_choice == "Logistic Regression":
                    if vectorizer and lr_model:
                        input_vec = vectorizer.transform([clean_input])
                        pred = lr_model.predict(input_vec)[0]
                        acc = lr_acc
                        pred_status = "REAL" if pred == 1 else "FAKE"
                        
                elif model_choice == "CNN":
                    if cnn_tokenizer and cnn_model:
                        seq = cnn_tokenizer.texts_to_sequences([clean_input])
                        pad_seq = pad_sequences(seq, maxlen=cnn_max_len, padding='post', truncating='post')
                        pred_prob = cnn_model.predict(pad_seq, verbose=0)[0][0]
                        pred = 1 if pred_prob >= 0.5 else 0
                        acc = cnn_acc
                        pred_status = "REAL" if pred == 1 else "FAKE"

                # --- Results Display ---
                st.write("---")
                
                st.markdown("<h3 style='color:#00FFFF;'>⚡️ Analysis Result:</h3>", unsafe_allow_html=True)
                
                if pred_status == "REAL":
                    st.success(f"🟢 **CONCLUSION:** The chosen model predicts this news is **REAL**! (Confidence: {round(acc*100,2)}%)")
                    st.balloons()
                elif pred_status == "FAKE":
                    st.error(f"🔴 **CONCLUSION:** The chosen model predicts this news is **FAKE**! (Confidence: {round(acc*100,2)}%)")
                    st.snow()
                else:
                    st.warning("Model prediction could not be performed due to a training error or missing data.")

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "📊 Model Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 Comprehensive Model Performance")
    
    st.markdown("<h3 style='color:#FF66CC;'>Key Accuracy Metrics</h3>", unsafe_allow_html=True)
    
    # Use a custom table/metrics for a better visual
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.metric(label="Random Forest Accuracy", value=f"{rf_acc*100:.2f}%", delta_color="normal")
    with colB:
        st.metric(label="Logistic Regression Accuracy", value=f"{lr_acc*100:.2f}%", delta_color="normal")
    with colC:
        st.metric(label="CNN Deep Learning Accuracy", value=f"{cnn_acc*100:.2f}%", delta_color="normal")
        
    st.write("---")
    
    st.markdown("<h3 style='color:#00FFFF;'>Detailed Classification Reports</h3>", unsafe_allow_html=True)

    with st.expander("Random Forest Classification Report"):
        if isinstance(rf_report, dict):
            report_df = pd.DataFrame(rf_report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)
        else:
            st.code(rf_report)

    with st.expander("Logistic Regression Classification Report"):
        if isinstance(lr_report, dict):
            report_df = pd.DataFrame(lr_report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)
        else:
            st.code(lr_report)
            
    # Note: CNN report is usually just accuracy unless a full prediction set is run.
    st.info("💡 **CNN Model:** The accuracy is reported, as the full classification report requires separate prediction steps.")

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "ℹ️ About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("ℹ️ About the Project: Technology Stack")
    st.markdown("""
    This advanced application is built upon a robust and modern technology stack:
    
    * **Core Library:** **Streamlit** (for the interactive, neon-themed Web UI).
    * **Deep Learning:** **TensorFlow/Keras** for the cutting-edge 1D Convolutional Neural Network (CNN).
    * **Machine Learning:** **Scikit-learn** for traditional ML models (Random Forest, Logistic Regression).
    * **Data Science:** **Pandas** for efficient data handling and preprocessing.
    * **Visualization:** **Plotly** (via Streamlit) for dynamic and interactive charts.
    
    ### ⚙️ How it Works:
    1.  **Preprocessing:** Text is cleaned, tokenized, and converted into numerical features (TF-IDF or padded sequences).
    2.  **Training:** Three distinct models are trained simultaneously for comparison.
    3.  **Prediction:** Your input text is processed and fed to the selected model to output a **REAL** or **FAKE** label.
    
    **👩‍💻 Developers:** Lakshya Pratap Singh
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align:center; color:#FF66CC;'>© 2025 Fake News Detector | Built with 💙 and ⚡️ using Streamlit (Neon Dark Mode)</p>", unsafe_allow_html=True)