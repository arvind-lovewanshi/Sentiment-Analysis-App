import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import base64
from PIL import Image

# Endpoint of Flask backend
prediction_endpoint = "http://127.0.0.1:5000/predict"

# Set page config
st.set_page_config(page_title="Text Sentiment Predictor", layout="wide")

# Custom CSS + banner image
st.markdown("""
    <style>
    .banner {
        background-image: url('https://images.unsplash.com/photo-1522199710521-72d69614c702');
        background-size: cover;
        background-position: center;
        padding: 100px 0;
        border-radius: 10px;
    }
    .banner-text {
        text-align: center;
        color: white;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
    }
    .subtext {
        text-align: center;
        color: white;
        font-size: 1.2em;
        margin-top: 10px;
        text-shadow: 1px 1px 2px #000000;
    }
    </style>

    <div class="banner">
        <div class="banner-text">Understand the Emotions Behind the Words 😊</div>
        <div class="subtext">Use machine learning to detect the sentiment of your text or CSV file.</div>
    </div>
    <br>
""", unsafe_allow_html=True)

# UI layout
col1, col2 = st.columns(2)

# Left = Bulk prediction
with col1:
    st.header("📄 Bulk Prediction (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Sentence' column", type="csv")

    if uploaded_file is not None and st.button("Predict on CSV"):
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)

        if response.status_code == 200:
            # Handle prediction CSV
            response_bytes = BytesIO(response.content)
            response_df = pd.read_csv(response_bytes)
            st.subheader("📋 Prediction Results")
            st.dataframe(response_df)

            # Download button
            st.download_button(
                label="📥 Download Predictions",
                data=response_bytes,
                file_name="Predictions.csv",
                key="csv_download"
            )

            # If graph exists in header
            if "X-Graph-Data" in response.headers:
                st.subheader("📊 Sentiment Distribution")
                graph_data = base64.b64decode(response.headers["X-Graph-Data"])
                st.image(graph_data, caption="Sentiment Distribution")
        else:
            st.error(f"❌ Server Error: {response.text}")

# Right = Single sentence prediction
with col2:
    st.header("💬 Predict Sentiment of a Sentence")
    user_input = st.text_area("Type your sentence here:", "")

    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            response = requests.post(prediction_endpoint, json={"text": user_input})

            if response.status_code == 200:
                try:
                    result = response.json()
                    prediction = result["prediction"]
                    emoji = "😊" if prediction == "Positive" else "😞"
                    st.success(f"**Predicted Sentiment:** {prediction} {emoji}")
                except Exception as e:
                    st.error(f"❌ Failed to parse response: {str(e)}")
            else:
                st.error(f"❌ Server Error: {response.text}")
        else:
            st.warning("⚠️ Please enter some text to predict.")

# Footer
st.markdown("""
    <hr>
    <center>Made with ❤️ using Streamlit & Flask</center>
""", unsafe_allow_html=True)
