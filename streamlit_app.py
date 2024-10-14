import pandas as pd
import string
import emoji
import demoji
import re
import contractions
import nltk
import pickle
import os
from wordcloud import WordCloud
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from detoxify import Detoxify
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from dotenv import load_dotenv
# load_dotenv()

api_keyy=st.secrets["GOOGLE_API_KEY"]
model_name=st.secrets["MODEL_NAME"]

genai.configure(api_key=api_keyy)
model = genai.GenerativeModel(model_name=model_name)

nltk.download('stopwords')

# Define your text processing functions
def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def remove_special_characters(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_newline(text):
    text = text.replace("\n", " ")
    return text

def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def stop_words_removal(query):
    stop = set(stopwords.words('english'))
    query = ' '.join([word for word in query.split() if word not in stop])
    return query

def token_text(query):
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_test_seq = tokenizer.texts_to_sequences([query])
    max_length = 500  
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    return X_test_pad

def preprocess(query):
    query = query.lower()
    query = remove_URL(query)
    query = remove_newline(query)
    query = remove_html(query)
    query = remove_non_ascii(query)
    query = remove_punct(query)
    query = stop_words_removal(query)
    query = token_text(query)
    return query

def make_prediction(query):
    preprocessed_query = preprocess(query)
    loaded_model = load_model('lstm_toxicity_model_epoch_8.keras')
    predictions_prob = loaded_model.predict(preprocessed_query)
    predictions = (predictions_prob > 0.5).astype(int)[0]
    return predictions, predictions_prob[0]

def extract_list(llm_generated_words):
    match = re.search(r'\[(.*?)\]', llm_generated_words)
    if match:
        array_string = match.group(0)
        array_list = eval(array_string)
        return array_list
    
    return ["Clean"]

def gemini_extract_word(inputtext, labels):
    # return """hi ["hello","hi"]"""
    print("me")
    print(inputtext)
    print("==============")
    print(labels)
    # inputtext="am good and super"
    system_message = """You’re a highly skilled text processing AI specializing in natural language understanding and extraction tasks. Your expertise lies in accurately identifying and extracting relevant words or phrases from given texts based on specified labels.
Your task is to extract words from a provided text that are related to a specified label and return the answer in a Python list format only, with no additional text or explanation.
Make sure to focus solely on the relevant words associated with the provided label, and return them in a list format like this: `["word1", "word2", ...]`.
Input Text:{}
Predicted Label:{}"""
    print("inside extraction")
    try:
        prompt=system_message.format(inputtext,labels)
        response = model.generate_content(prompt,
                                      safety_settings=[
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ])
        print("outside extraction")
        return response.text
    except Exception as e:
        print("")
        return """We apologize for the inconvenience. There was an issue while communicating with the Google API for WordCloud. 
              Please try again or retry after some time. Thank you for your understanding."""

def gemini_explanation(inputtext,labels):
    # return "hi"
    system_message = """ You’re a knowledgeable AI model evaluator with extensive experience in analyzing prediction models and their outputs. Your expertise lies in providing clear, insightful explanations for why a model has assigned a particular labels (labels are separated by commas) to a specific text, considering factors such as context, keywords, and overall sentiment.
Your task is to explain the reasoning behind a label assigned by your prediction model. 
NOTE **only extract the word from given input text is really mean to the given labels** 
Explain Detailly
Input Text:{}
Predicted Label:{}
"""
    try:
        print("inside explanation")
        prompt=system_message.format(inputtext,labels)
        response = model.generate_content(prompt,
                                      safety_settings=[
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ])
        print("outside explanation")
        return response.text
    except Exception as e:
        print("")
        return """We apologize for the inconvenience. There was an issue while communicating with the Google API for LLM Explanation. 
              Please try again or retry after some time. Thank you for your understanding."""

def extract_words_llm(query, llm_labels, prediction):
    if 1 not in prediction:
        return "NO toxic in text, it's clean!",["Clean"]
    
    llm_explanation = gemini_explanation(query, llm_labels)
    llm_extracted_words = gemini_extract_word(query, llm_labels)
    word_list=extract_list(llm_extracted_words)
    return llm_explanation, word_list

def generate_wordcloud(words):
    text = ' '.join(words)
    wordcloud = WordCloud(color_func=lambda *args, **kwargs: "red").generate(text)
    wordcloud_path = "wordcloud_red.png"
    wordcloud.to_file(wordcloud_path) 
    return wordcloud_path

# Streamlit App
def main():
    st.title("Comment Toxicity Analyzer")

    query = st.text_area("Enter comment to analyze:")

        # Model selection
    model_options = ["LSTM Model", "Pretrained Bert Detoxify"]
    selected_model = st.selectbox("Please Choose a Model:", model_options)

    if st.button("Analyze"):
        labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

        if selected_model=="Pretrained Bert Detoxify":
            print("entering BERT Detoxify")
            results = Detoxify('original').predict(query)
            labels_mapping = {'toxicity': 'Toxic','severe_toxicity': 'Severe Toxic','obscene': 'Obscene','threat': 'Threat','insult': 'Insult','identity_attack': 'Identity Hate'}
            formatted_results = {}
            for key, value in results.items():
                label = labels_mapping[key]
                formatted_results[label] = [
                    0 if value <= 0.5 else 1, 
                    f"{int(value * 100)}%"]
                
                llm_labels = ""
                detoxify_prediction=[]
                for i, (label, result) in enumerate(formatted_results.items()):
                    detoxify_prediction.append(result[0])
                    if result[0] == 1: 
                        llm_labels += label + ","
            if 1 not in detoxify_prediction:
                formatted_results["Clean"]=[1,"100%"]
            print("hi")
            print(formatted_results)
            llm_explanation, llm_extracted_words = extract_words_llm(query, llm_labels, detoxify_prediction)
            labels = list(formatted_results.keys())
            predictions = [result[0] for result in formatted_results.values()]
            probabilities = [int(result[1][:-1].replace("%","")) for result in formatted_results.values()]  # Convert percentage string to int
            print(probabilities)
          

            bar_colors = []
            for i, pred in enumerate(predictions):
                if labels[i] == "Clean" and pred == 1:
                    bar_colors.append('green')  
                elif pred == 1:
                    bar_colors.append('red') 
                else:
                    bar_colors.append('green') 

         
            fig, ax = plt.subplots(figsize=(8, 5))  
            bars = ax.barh(labels, probabilities, color=bar_colors, alpha=0.9, edgecolor='black')
            for bar in bars:
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width()}%", va='center', fontsize=10, color='black', fontweight='bold')
            ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Labels', fontsize=12, fontweight='bold')
            ax.set_title('Toxicity Predictions', fontsize=14, fontweight='bold', color='darkblue')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            st.subheader("Toxicity Predictions Distribution")
            st.pyplot(fig)

            st.subheader("Formatted Predictions")
            prediction_df = pd.DataFrame(formatted_results).T
            prediction_df.columns = ["Prediction", "Probability"]
            prediction_df["Probability"] = prediction_df["Probability"].str.replace('%', '').astype(int)  # Convert to int for sorting
            prediction_df = prediction_df.sort_values(by="Probability", ascending=False)
            def color_coding(row):
                return ['background-color:lightgray'] * len(
                    row) if row.Prediction == 1 else ['background-color:white'] * len(row)
            st.dataframe(prediction_df.style.apply(color_coding, axis=1),use_container_width=True)
  

            json_results = {label: {'Prediction': pred[0], 'Probability': pred[1]} for label, pred in formatted_results.items()}
            with st.expander("Show Results in JSON", expanded=False):
                st.json(json_results)
            st.subheader("LLM Explanation")
            st.write(llm_explanation)
        
            if llm_extracted_words:
                wordcloud_path = generate_wordcloud(llm_extracted_words)
                st.subheader("Word Cloud of Extracted Words")
                st.image(wordcloud_path, caption="")


            # st.title("Download Existing PDF")
            # pdf_file_path = "mercury-CertificateCompletionReport.pdf"  
            # with open(pdf_file_path, "rb") as pdf_file:
            #     st.download_button(
            #         label="Download PDF",
            #         data=pdf_file,
            #         file_name="Report.pdf", 
            #         mime="application/pdf"
            #     )

        else:   
            print("entering LSTM")
            prediction, predictions_prob = make_prediction(query)
            formatted_results = {labels[i]: [prediction[i], f"{int(predictions_prob[i] * 100)}%"] for i in range(len(labels))}
            
            llm_labels = ""
            detoxify_prediction=[]
            for i, (label, result) in enumerate(formatted_results.items()):
                detoxify_prediction.append(result[0])
                if result[0] == 1: 
                    llm_labels += label + ","
            if 1 not in detoxify_prediction:
                formatted_results["Clean"]=[1,"100%"]
   

            print(formatted_results)
            llm_explanation, llm_extracted_words = extract_words_llm(query, llm_labels, detoxify_prediction)
            labels = list(formatted_results.keys())
            predictions = [result[0] for result in formatted_results.values()]
            probabilities = [int(result[1][:-1].replace("%","")) for result in formatted_results.values()]  # Convert percentage string to int
            print(probabilities)


            bar_colors = []
            for i, pred in enumerate(predictions):
                if labels[i] == "Clean" and pred == 1:
                    bar_colors.append('green')  
                elif pred == 1:
                    bar_colors.append('red') 
                else:
                    bar_colors.append('green') 

         
            fig, ax = plt.subplots(figsize=(8, 5))  
            bars = ax.barh(labels, probabilities, color=bar_colors, alpha=0.9, edgecolor='black')
            for bar in bars:
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width()}%", va='center', fontsize=10, color='black', fontweight='bold')
            ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Labels', fontsize=12, fontweight='bold')
            ax.set_title('Toxicity Predictions', fontsize=14, fontweight='bold', color='darkblue')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            st.subheader("Toxicity Predictions Distribution")
            st.pyplot(fig)

            st.subheader("Formatted Predictions")
            prediction_df = pd.DataFrame(formatted_results).T
            prediction_df.columns = ["Prediction", "Probability"]
            prediction_df["Probability"] = prediction_df["Probability"].str.replace('%', '').astype(int)  # Convert to int for sorting
            prediction_df = prediction_df.sort_values(by="Probability", ascending=False)
            def color_coding(row):
                return ['background-color:lightgray'] * len(
                    row) if row.Prediction == 1 else ['background-color:white'] * len(row)
            st.dataframe(prediction_df.style.apply(color_coding, axis=1),use_container_width=True)
  

            json_results = {label: {'Prediction': pred[0], 'Probability': pred[1]} for label, pred in formatted_results.items()}
            with st.expander("Show Results in JSON", expanded=False):
                st.json(json_results)
            st.subheader("LLM Explanation")
            st.write(llm_explanation)
        
            if llm_extracted_words:
                wordcloud_path = generate_wordcloud(llm_extracted_words)
                st.subheader("Word Cloud of Extracted Words")
                st.image(wordcloud_path, caption="")


            # st.title("Download Existing PDF")
            # pdf_file_path = "mercury-CertificateCompletionReport.pdf"  
            # with open(pdf_file_path, "rb") as pdf_file:
            #     st.download_button(
            #         label="Download PDF",
            #         data=pdf_file,
            #         file_name="Report.pdf", 
            #         mime="application/pdf"
            #     )
                

if __name__ == "__main__":
    main()
