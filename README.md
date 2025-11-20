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


<img width="1600" height="787" alt="image" src="https://github.com/user-attachments/assets/9d977f04-47a1-4b4b-907e-82c17ac8b83d" />
<img width="1600" height="789" alt="image" src="https://github.com/user-attachments/assets/fe787c85-2ff3-41f8-9b24-80962099b353" />
<img width="1600" height="785" alt="image" src="https://github.com/user-attachments/assets/a7e8b648-bcb6-45b8-b2f3-438220dd4b14" />
<img width="1600" height="793" alt="image" src="https://github.com/user-attachments/assets/71458abf-b6f9-4fcd-abdd-5d17d3b2dec5" />




