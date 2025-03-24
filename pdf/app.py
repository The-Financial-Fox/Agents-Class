import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdfplumber

from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(
    page_title="AI PDF Extractor & FP&A Assistant",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("AI-Powered PDF Data Extractor & FP&A Assistant")

# File uploader for PDF
uploaded_files = st.file_uploader(
    "Upload one or more PDF files to extract data",
    type=["pdf"],
    accept_multiple_files=True
)

# Button to process PDFs
if st.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

    # Collect text from each PDF
    all_extracted_texts = []
    for uploaded_file in uploaded_files:
        pdf_file_name = uploaded_file.name
        st.write(f"**Processing:** {pdf_file_name}")

        # Use pdfplumber to extract text
        with pdfplumber.open(uploaded_file) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        
        # Optionally, store or display the extracted text
        all_extracted_texts.append({"filename": pdf_file_name, "text": text_content})

    # Convert extracted text to a DataFrame for easy handling
    df_extracted = pd.DataFrame(all_extracted_texts)

    st.write("### Extracted Text DataFrame")
    st.dataframe(df_extracted)

    # **Optional**: Send the combined PDF text to Groq for AI-based analysis
    # Feel free to tailor the prompt as needed for your use case
    st.subheader("🤖 AI-Generated Insights from PDFs")
    client = Groq(api_key=GROQ_API_KEY)

    # Combine all text into one chunk (adjust as needed)
    combined_text = "\n\n".join([item["text"] for item in all_extracted_texts])

    prompt = f"""
    You are an AI agent that reviews text from multiple PDF documents.
    Extract important insights, summarize key points, and provide actionable recommendations.
    The combined text from the PDFs is below:

    {combined_text}
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an advanced AI specializing in reading PDF text for analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",
        )
        ai_commentary = response.choices[0].message.content
    except Exception as e:
        ai_commentary = f"Error calling Groq API: {e}"

    # **Display AI Commentary**
    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.subheader("📖 AI-Generated Commentary")
    st.write(ai_commentary)
    st.markdown('</div>', unsafe_allow_html=True)

    # **Save DataFrame to Excel**
    output_excel_file = "pdf_extracted_data.xlsx"
    df_extracted.to_excel(output_excel_file, index=False)
    st.success(f"Data saved to {output_excel_file}!")

    # Provide a download button
    with open(output_excel_file, "rb") as f:
        st.download_button(
            label="Download Excel File",
            data=f,
            file_name=output_excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
