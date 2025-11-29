## Cyberattack Detection System
A comprehensive Streamlit dashboard that detects multiple types of cyberattacks using machine learning, deep learning, and NLP-based models.
This system provides real-time predictions for email phishing, malicious URLs, and DDoS attacks.

# Phishing Email Detection (Bi-LSTM)
- Processes both the email subject and body text
- Cleans and tokenizes text using NLTK
- Uses a Bidirectional LSTM model to classify emails as Phishing or Not Phishing

# Phishing URL Detection (Logistic Regression + NLP)
- Cleans and normalizes URLs
- Converts URLs into vector form using CountVectorizer
- Logistic Regression model predicts whether a URL is Malicious or Safe

# DDoS Attack Detection (Neural Network)
- Loads SDN network traffic dataset
- Extracts numerical features such as packet counts, durations, byte counts, etc.
- Scales features using StandardScaler
- Trains a fully connected neural network to classify Normal vs DDoS traffic

**# Run Locally 🚀**
- streamlit run app.py

**Output**


<img width="1600" height="787" alt="image" src="https://github.com/user-attachments/assets/9d977f04-47a1-4b4b-907e-82c17ac8b83d" />
<img width="1600" height="789" alt="image" src="https://github.com/user-attachments/assets/fe787c85-2ff3-41f8-9b24-80962099b353" />
<img width="1600" height="785" alt="image" src="https://github.com/user-attachments/assets/a7e8b648-bcb6-45b8-b2f3-438220dd4b14" />
<img width="1600" height="793" alt="image" src="https://github.com/user-attachments/assets/71458abf-b6f9-4fcd-abdd-5d17d3b2dec5" />




