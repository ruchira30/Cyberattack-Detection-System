**Cyberattack Detection System**

A Streamlit dashboard that detects different types of cyberattacks using machine learning and deep learning models.

 **Features**
1.**Phishing Email Detection (Bi-LSTM)**

- Processes subject + message text

- Uses a Bi-LSTM model to classify emails as Phishing or Not Phishing

2. **Phishing URL Detection (NLP + Logistic Regression)**

- Cleans and vectorizes URLs

- Logistic Regression model predicts whether a URL is Malicious or Safe

3. **DDoS Attack Detection (Neural Network)**

- Loads SDN network traffic dataset

- Neural network classifies normal vs DDoS traffic

- Includes visualizations: protocol counts, packet/byte histograms, class distribution

**Streamlit Dashboard 🖥️**

The dashboard includes:

- Scan Email
- Scan URL
- Scan DDoS
Each section provides real-time predictions through a simple interface.

🛠️ **Technologies Used**

- Python

- Streamlit

- TensorFlow / Keras

- Scikit-Learn

- NLTK

- Pandas / NumPy

- Matplotlib / Seaborn

  
**Run Locally 🚀**

pip install -r requirements.txt

streamlit run app.py

**Output**
1. Phishing Email


2. Phishing URLs


3. DDos attacks 
