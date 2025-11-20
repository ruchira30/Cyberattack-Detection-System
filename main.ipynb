import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import nltk
import streamlit as st

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = nltk.stem.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess input text."""
    if isinstance(text, str):  # Ensure input is string
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.lower()  # Convert to lowercase
        text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
        text = ' '.join(stemmer.stem(word) for word in text.split())  # Apply stemming
        return text
    return ''  # Return empty string if input is not valid

# Function to load and preprocess data
def load_and_preprocess_data():
    # Load the dataset
    url = 'datasets/phishing_data_by_type.csv'
    data = pd.read_csv(url)

    # Apply preprocessing to 'Subject' and 'Text' columns
    data['Subject'] = data['Subject'].apply(preprocess_text)
    data['Text'] = data['Text'].apply(preprocess_text)
    data['Combined'] = data['Subject'] + ' ' + data['Text']

    # Define features and target variable
    X = data['Combined']
    y = data['Type'].apply(lambda x: 1 if x == 'Fraud' or x == 'Phishing' else 0)

    # Tokenization and sequence padding
    max_words = 5000  # Vocabulary size
    maxlen = 200  # Sequence length
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)

    # Convert text to sequences
    X_seq = tokenizer.texts_to_sequences(X)
    X_seq = pad_sequences(X_seq, padding='post', maxlen=maxlen)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    return tokenizer, maxlen, X_train, X_test, y_train, y_test

# Function to define and compile the Bi-LSTM model
def create_model(max_words, maxlen):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=maxlen),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(X_train, y_train, X_test, y_test):
    model = create_model(max_words=5000, maxlen=200)  # Create the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

# Function to make predictions
def predict_phishing(text, model, tokenizer, maxlen):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    return "Phishing" if prediction[0][0] > 0.5 else "Not Phishing"

# Preprocessing function for URLs
def preprocess_url(url):
    """Clean and preprocess input URL."""
    if isinstance(url, str):
        # Lowercase the URL
        url = url.lower()
        # Remove protocol (http, https, ftp, etc.)
        url = re.sub(r'^(http://|https://|ftp://|www\.)', '', url)
        # Remove trailing slashes
        url = url.rstrip('/')
        return url
    return ''

# Load and preprocess URL data
def load_url_data():
    # Load the dataset with URLs and labels
    url = 'datasets/phishing_site_urls.csv'  # Path to your CSV file
    data = pd.read_csv(url)

    # Ensure data contains only necessary columns
    data = data[['URL', 'Label']]

    # Convert labels to binary: 'bad' as 1 and anything else as 0
    data['Label'] = data['Label'].apply(lambda x: 1 if x == 'bad' else 0)

    # Preprocess URLs
    data['URL'] = data['URL'].apply(preprocess_url)

    return data['URL'], data['Label']

# Function to train Logistic Regression model for URL detection
def train_logistic_regression_model(urls, labels, progress_bar):
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(urls)
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, labels, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    model.fit(X_train, y_train)  # Fit the model to the training data

    # Simulate training steps for progress bar
    for i in range(1, 6):  # Simulating training steps
        progress = i / 5
        progress_bar.progress(progress)  # Update progress bar

    return model, vectorizer

def predict_phishing_url(url, model, vectorizer):
    url_processed = preprocess_url(url)
    url_vector = vectorizer.transform([url_processed])
    prediction = model.predict(url_vector)

    # Debugging: Show the predicted value and the raw prediction probability
    prediction_proba = model.predict_proba(url_vector)

    # Print out the prediction details for debugging
    print(f"URL: {url}, Prediction: {prediction[0]}, Probability: {prediction_proba[0]}")

    return "Phishing" if prediction[0] == 1 else "Not Phishing"


def load_ddos_data():
    """Load and preprocess the DDoS dataset."""
    try:
        # Load dataset
        data = pd.read_csv("datasets/dataset_sdn.csv")

        features = ['pktcount', 'bytecount', 'dur', 'tot_dur', 'flows', 'pktrate', 'port_no', 'tx_bytes', 'rx_bytes']
        target = 'label'

        if target not in data.columns:
            st.error(f"Target column '{target}' not found in the dataset.")
            return None, None, data

        # One-Hot Encoding for categorical features, limiting the number of categories to reduce memory
        encoder = OneHotEncoder(sparse_output=True, drop='first', max_categories=100)  # limit categories
        encoded_features = encoder.fit_transform(data[features])

        # Prepare features and target variable
        X = encoded_features
        y = data[target].values

        # Convert labels to categorical
        y = to_categorical(y)

        return X, y, data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def train_ddos_model(X, y):
    """Train the DDoS detection model and return predictions."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define a simple neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))  # Use softmax for multi-class classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train.toarray(), y_train, epochs=30, batch_size=50, verbose=1)  # convert sparse to array for training

    # Evaluate the model
    y_pred_probs = model.predict(X_test.toarray())  # convert sparse to array for prediction
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get the class with highest probability
    y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels back to single class

    return y_test_labels, y_pred

def display_data_visualizations(data):
    """Display data visualizations including a quadrant of four graphs."""
    st.subheader("Data Visualizations")

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Count of different protocols
    protocol_counts = data['Protocol'].value_counts()
    axs[0, 0].bar(protocol_counts.index, protocol_counts.values, color='skyblue')
    axs[0, 0].set_title('Count of Different Protocols')
    axs[0, 0].set_xlabel('Protocols')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].tick_params(axis='x', rotation=45)

    # Distribution of packet counts
    axs[0, 1].hist(data['pktcount'], bins=30, color='salmon', edgecolor='black')
    axs[0, 1].set_title('Distribution of Packet Counts')
    axs[0, 1].set_xlabel('Packet Count')
    axs[0, 1].set_ylabel('Frequency')

    # Distribution of byte counts
    axs[1, 0].hist(data['bytecount'], bins=30, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title('Distribution of Byte Counts')
    axs[1, 0].set_xlabel('Byte Count')
    axs[1, 0].set_ylabel('Frequency')

    # Class Distribution
    class_counts = data['label'].value_counts()
    colors = ['orange', 'blue']  # Specify colors for class 0 and class 1
    class_labels = ['Not DDoS', 'DDoS']  # Define your class labels

    # Create bar chart with labels
    axs[1, 1].bar(class_labels, class_counts.values, color=colors)
    axs[1, 1].set_title('Class Distribution')
    axs[1, 1].set_xlabel('Classes')
    axs[1, 1].set_ylabel('Count')


    # Adjust layout
    plt.tight_layout()
    plt.show()
    # Display the plots in Streamlit
    st.pyplot(fig)


def display_classification_report(y_test, y_pred):
    """Display a simple classification report and confusion matrix in Streamlit."""

    # Display the classification report as plain text
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)  # Display as plain text to avoid DataFrame styling issues

    # Display the confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    st.pyplot(fig)
# Streamlit app interface
def main():
    st.set_page_config(page_title="Cyberattack Detection System", layout="wide")  # Set layout to wide
    st.title("Cyberattack Detection System")

    # Sidebar navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select an option:", ["Select", "Scan Emails", "Scan URLs", "Scan DDoS"])

    # Initialize variables for URL detection model
    url_model = None
    vectorizer = None

    if option == "Select":
        st.image('cyber-attack-small.png', use_column_width='auto')
        st.write("")
    # Check if the user selected "Scan Emails"
    model = None
    tokenizer = None
    maxlen = None

    # Check if the user selected "Scan Emails"
    if option == "Scan Emails":
        # Load and preprocess data only when "Scan Emails" is selected
        tokenizer, maxlen, X_train, X_test, y_train, y_test = load_and_preprocess_data()

        # Train the model only when the Predict button is clicked
        st.subheader("Phishing Detection")
        st.write("Enter email subject or body text to predict if it's phishing.")

        # Text input
        user_input = st.text_area("Input", placeholder="Enter email subject or text here...")

        if st.button("Predict"):
            # Show a loading spinner while predicting
            with st.spinner("Training model and scanning email for phishing..."):
                # Train the model if not already done
                if model is None:
                    model = train_model(X_train, y_train, X_test, y_test)
                # Run the prediction
                prediction = predict_phishing(user_input, model, tokenizer, maxlen)
                st.success(f"The prediction is: {prediction}")


    # Check if the user selected "Scan URLs"
    elif option == "Scan URLs":
        # Load and preprocess URL data only when "Scan URLs" is selected
        urls, labels = load_url_data()

        st.subheader("URL Phishing Detection")
        st.write("Enter a URL to check if it is phishing.")

        # Text input for URL
        url_input = st.text_input("URL", placeholder="Enter URL here...")

        # Train and use Logistic Regression model
        if st.button("Scan URL"):
            with st.spinner("Training model and scanning URL for phishing..."):
                # Train Logistic Regression model if not already done
                if url_model is None:
                    progress_bar = st.progress(0)  # Initialize the progress bar
                    url_model, vectorizer = train_logistic_regression_model(urls, labels, progress_bar)

                # Make prediction using the Logistic Regression model
                prediction = predict_phishing_url(url_input, url_model, vectorizer)
                st.success(f"Prediction: {prediction}")


    elif option == "Scan DDoS":
        st.subheader("DDoS Detection")

        if st.button("Analyze"):
            with st.spinner("Loading and preprocessing DDoS data..."):
                X, y, data = load_ddos_data()
                if X is not None and y is not None and data is not None:
                    display_data_visualizations(data)
                    y_test, y_pred = train_ddos_model(X, y)
                    display_classification_report(y_test, y_pred)
                else:
                    st.error("Failed to load DDoS data.")

if __name__ == "__main__":
    main()


