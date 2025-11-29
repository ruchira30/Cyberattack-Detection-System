import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Phishing Emails
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = ' '.join(word for word in text.split() if word not in stop_words)
        text = ' '.join(stemmer.stem(word) for word in text.split())
        return text
    return ''

def load_and_preprocess_data():
    url = 'dataset/phishing_data_by_type.csv'
    data = pd.read_csv(url)
    data['Subject'] = data['Subject'].apply(preprocess_text)
    data['Text'] = data['Text'].apply(preprocess_text)
    data['Combined'] = data['Subject'] + ' ' + data['Text']
    X = data['Combined']
    y = data['Type'].apply(lambda x: 1 if x == 'Fraud' or x == 'Phishing' else 0)
    max_words = 5000
    maxlen = 200
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_seq = pad_sequences(X_seq, padding='post', maxlen=maxlen)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    return tokenizer, maxlen, X_train, X_test, y_train, y_test

def create_model(max_words, maxlen):
    """ Bi-LSTM Model """
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=maxlen),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test):
    model = create_model(max_words=5000, maxlen=200)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

def predict_phishing(text, model, tokenizer, maxlen):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    return "Phishing" if prediction[0][0] > 0.5 else "Not Phishing"

# URL Phishing 

def preprocess_url(url):
    if not isinstance(url, str):
        return ''
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)    
    url = url.rstrip('/')
    return url

def load_url_data():
    file1 = 'dataset/phishing_url_data_1.csv'
    file2 = 'dataset/phishing_urls_data.csv'
    df1 = pd.read_csv(file1)[['URL', 'Label']]
    df2 = pd.read_csv(file2)[['URL', 'Label']]
    data = pd.concat([df1, df2], ignore_index=True)
    data['URL'] = data['URL'].apply(preprocess_url)
    data['Label'] = (
        data['Label']
        .astype(str)
        .str.lower()
        .map({'bad': 1, 'good': 0})
    )
    data = data.dropna(subset=['Label'])
    return data['URL'], data['Label'].astype(int)

def train_logistic_regression_model(urls, labels, progress_bar):
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(urls)
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000) 
    model.fit(X_train, y_train) 
    for i in range(1, 6):  
        progress = i / 5
        progress_bar.progress(progress)
    return model, vectorizer

def predict_phishing_url(url, model, vectorizer):
    url_processed = preprocess_url(url)
    url_vector = vectorizer.transform([url_processed])
    prediction = model.predict(url_vector)
    prediction_proba = model.predict_proba(url_vector)
    print(f"URL: {url}, Prediction: {prediction[0]}, Probability: {prediction_proba[0]}")
    return "Phishing" if prediction[0] == 1 else "Not Phishing"

# DDoS Detection
def load_ddos_data():
    try:
        data = pd.read_csv("dataset/dataset_sdn.csv")
        features = ['pktcount', 'bytecount', 'dur', 'tot_dur','flows', 'pktrate', 'port_no', 'tx_bytes', 'rx_bytes']
        target = 'label'
        if not all(f in data.columns for f in features):
            st.error("One or more required columns are missing from dataset.")
            return None, None, None,None
        if target not in data.columns:
            st.error(f"Target column '{target}' not found.")
            return None, None, None,None
        X = data[features].values
        y = data[target].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, data, scaler
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def train_ddos_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    return y_test, y_pred, model

def display_data_visualizations(data):
    st.subheader("Data Visualizations")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    protocol_counts = data['Protocol'].value_counts()
    axs[0, 0].bar(protocol_counts.index, protocol_counts.values, color='skyblue')
    axs[0, 0].set_title('Count of Different Protocols')
    axs[0, 0].set_xlabel('Protocols')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 1].hist(data['pktcount'], bins=30, color='salmon', edgecolor='black')
    axs[0, 1].set_title('Distribution of Packet Counts')
    axs[0, 1].set_xlabel('Packet Count')
    axs[0, 1].set_ylabel('Frequency')
    axs[1, 0].hist(data['bytecount'], bins=30, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title('Distribution of Byte Counts')
    axs[1, 0].set_xlabel('Byte Count')
    axs[1, 0].set_ylabel('Frequency')
    class_counts = data['label'].value_counts()
    colors = ['orange', 'blue']
    class_labels = ['Not DDoS', 'DDoS']
    axs[1, 1].bar(class_labels, class_counts.values, color=colors)
    axs[1, 1].set_title('Class Distribution')
    axs[1, 1].set_xlabel('Classes')
    axs[1, 1].set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig)

def display_classification_report(y_test, y_pred):
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    st.pyplot(fig)


# Streamlit
def main():
    st.set_page_config(page_title="Cyberattack Detection System", layout="wide")
    st.title("Cyberattack Detection System")
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select an option:", ["Select", "Scan Emails", "Scan URLs", "Scan DDoS"])

    if "email_model" not in st.session_state:
        st.session_state.email_model = None
        st.session_state.tokenizer = None
        st.session_state.maxlen = None
    if "url_model" not in st.session_state:
        st.session_state.url_model = None
        st.session_state.vectorizer = None

    if option == "Select":
        st.image('dataset/images.jpg', use_container_width=True)
        st.write("Select an option from the sidebar to start detection.")

    elif option == "Scan Emails":
        st.subheader("Phishing Email Detection")
        user_input = st.text_area("Enter email subject or body text:")
        if st.button("Predict Email"):
            if not user_input.strip():
                st.warning("Please enter email text to scan.")
            else:
                if st.session_state.email_model is None or st.session_state.tokenizer is None:
                    with st.spinner("Loading data and training model..."):
                        tokenizer, maxlen, X_train, X_test, y_train, y_test = load_and_preprocess_data()
                        st.session_state.tokenizer = tokenizer
                        st.session_state.maxlen = maxlen
                        st.session_state.email_model = train_model(X_train, y_train, X_test, y_test)
                with st.spinner("Scanning email for phishing..."):
                    result = predict_phishing(user_input, st.session_state.email_model, st.session_state.tokenizer, st.session_state.maxlen)
                    st.success(f"Prediction: {result}")

    elif option == "Scan URLs":
        st.subheader("URL Phishing Detection")
        url_input = st.text_input("Enter URL:")
        if st.button("Scan URL"):
            if not url_input.strip():
                st.warning("Please enter a URL to scan.")
            else:
                if st.session_state.url_model is None or st.session_state.vectorizer is None:
                    with st.spinner("Loading URL data and training model..."):
                        urls, labels = load_url_data()
                        progress_bar = st.progress(0)
                        st.session_state.url_model, st.session_state.vectorizer = train_logistic_regression_model(urls, labels, progress_bar)
                with st.spinner("Scanning URL for phishing..."):
                    result = predict_phishing_url(url_input, st.session_state.url_model, st.session_state.vectorizer)
                    st.success(f"Prediction: {result}")

    elif option == "Scan DDoS":
        st.subheader("DDoS Detection")
        if st.button("Analyze"):
            with st.spinner("Loading DDoS data..."):
                X, y, data, scaler = load_ddos_data()
            if X is not None:
                st.subheader("Network Traffic Visualizations")
                display_data_visualizations(data)
                with st.spinner("Training DDoS detection model..."):
                    y_test, y_pred, ddos_model = train_ddos_model(X, y)
                st.subheader("DDoS Classification Report")
                display_classification_report(y_test, y_pred)
            else:
                st.error("Failed to load DDoS data.")

if __name__ == "__main__":
    main()



