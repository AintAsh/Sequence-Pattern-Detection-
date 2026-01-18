import streamlit as st
import requests
import time

API_URL = "https://sequence-pattern-detection.onrender.com/predict"

# ---------- Page config ----------
st.set_page_config(
    page_title="Pattern Detection",
    page_icon="📈",
    layout="centered"
)
#-----------------------------------------

def explain_pattern(seq):
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]

    if all(d == diffs[0] for d in diffs):
        if diffs[0] > 0:
            return "The sequence shows a consistent increasing trend."
        elif diffs[0] < 0:
            return "The sequence shows a consistent decreasing trend."
        else:
            return "All values in the sequence are constant."

    if all(seq[i] == seq[i % 2] for i in range(len(seq))):
        return "The sequence follows an alternating pattern."

    return "No clear numerical pattern is detected in this sequence."


#------------------------------------------
# ---------- DARK THEME REFINEMENT ----------
st.markdown("""
<style>
.stApp {
    background-color: #0B0F14;
}

.card {
    background-color: #111827;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #1F2937;
    margin-bottom: 24px;
}

.title {
    font-size: 34px;
    font-weight: 700;
    color: #F9FAFB;
}
.subtitle {
    font-size: 15px;
    color: #9CA3AF;
    margin-bottom: 28px;
}

.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 12px;
}

label {
    color: #D1D5DB !important;
}

input {
    background-color: #020617 !important;
    color: #F9FAFB !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
}

button[kind="primary"] {
    background-color: #3B82F6 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 600;
    height: 48px;
}

.result-box {
    background-color: #020617;
    border-radius: 12px;
    padding: 18px;
    border-left: 5px solid #3B82F6;
    margin-top: 16px;
    color: #F9FAFB;
}

.confidence-high { color: #60A5FA; font-weight: 600; }
.confidence-mid  { color: #9CA3AF; font-weight: 600; }
.confidence-low  { color: #6B7280; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">Pattern Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detect numerical patterns using a trained RNN model</div>',
    unsafe_allow_html=True
)

# ---------- INPUT CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Input Sequence (exactly 5 numbers)</div>', unsafe_allow_html=True)
st.caption("Type numbers directly. Order matters.")

c1, c2, c3, c4, c5 = st.columns(5)

with c1: n1 = st.text_input("N1", "0")
with c2: n2 = st.text_input("N2", "1")
with c3: n3 = st.text_input("N3", "0")
with c4: n4 = st.text_input("N4", "1")
with c5: n5 = st.text_input("N5", "0")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICT ----------
predict = st.button("Detect Pattern", use_container_width=True)

# ---------- RESULT ----------
if predict:
    try:
        sequence = [int(n1), int(n2), int(n3), int(n4), int(n5)]

        # 🔄 subtle animation
        with st.spinner("Analyzing sequence..."):
            time.sleep(0.4)
            response = requests.post(API_URL, json={"sequence": sequence})
            result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]
        confidence = probability if prediction == 1 else (1 - probability)

        # ---------- RESULT CARD ----------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        label = "Pattern Detected" if prediction == 1 else "No Pattern Detected"

        st.markdown(
            f"""
            <div class="result-box">
                <b>{label}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- CONFIDENCE METER ----------
        if confidence >= 0.8:
            conf_label = "High confidence"
            conf_class = "confidence-high"
        elif confidence >= 0.6:
            conf_label = "Medium confidence"
            conf_class = "confidence-mid"
        else:
            conf_label = "Low confidence"
            conf_class = "confidence-low"

        st.markdown(
            f"<span class='{conf_class}'>Confidence: {confidence:.2%} ({conf_label})</span>",
            unsafe_allow_html=True
        )
        st.progress(confidence)

        # ---------- SEQUENCE VISUALIZATION ----------
        st.markdown('<div class="section-title">Sequence Visualization</div>', unsafe_allow_html=True)
        st.line_chart(sequence)

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- PATTERN EXPLANATION ----------
        explanation = explain_pattern(sequence)

        st.markdown('<div class="section-title">Pattern Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-box">
                {explanation}
            </div>
            """,
            unsafe_allow_html=True
        )




    except ValueError:
        st.error("Please enter valid integers only.")
    except Exception:
        st.error("API not reachable. Make sure FastAPI is running.")


# ---------- FOOTER ----------
st.markdown("""
<hr style="border: 1px solid #1F2937; margin-top: 40px;">

<div style="
    text-align: center;
    color: #6B7280;
    font-size: 13px;
    margin-bottom: 10px;
">
    Built with <b>TensorFlow</b>, <b>FastAPI</b>, and <b>Streamlit</b><br>
    Sequence Pattern Detection using Recurrent Neural Networks</b><br>
    Built by Om 
</div>
""", unsafe_allow_html=True)