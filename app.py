import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class DiseasePredictor:
    def __init__(self):
        # Placeholder for a disease dataset
        # Replace this with a real medical dataset for training
        self.symptoms = ["fever", "cough", "headache", "fatigue", "nausea"]
        self.diseases = ["Common Cold", "Flu", "Migraine", "COVID-19", "Food Poisoning"]
        self.vectorizer = TfidfVectorizer()
        self.clf = MultinomialNB()

        # Train the disease predictor model (replace with your real training data)
        X_train = self.vectorizer.fit_transform(self.symptoms)
        y_train = np.array(self.diseases)
        self.clf.fit(X_train, y_train)

    def predict_disease(self, symptom_description):
        # This is a placeholder function; replace it with your actual disease prediction logic
        # For simplicity, it uses a basic TF-IDF + Naive Bayes model
        X_test = self.vectorizer.transform([symptom_description])
        predicted_disease = self.clf.predict(X_test)
        return predicted_disease[0]

# Create an instance of the DiseasePredictor
disease_predictor = DiseasePredictor()

# Streamlit app
def run_medical_chat():
    st.title("HealthBot - Medical Chatbot")
    st.write("Describe your symptoms, and HealthBot will provide a potential diagnosis.")

    user_input = st.text_input("You:")
    
    if st.button("Get Diagnosis"):
        if user_input:
            disease_prediction = disease_predictor.predict_disease(user_input.lower())
            st.success(f"Based on the symptoms you described, it may be {disease_prediction}. Please consult with a healthcare professional for accurate diagnosis.")
        else:
            st.warning("Please enter symptoms.")

if __name__ == "__main__":
    run_medical_chat()
