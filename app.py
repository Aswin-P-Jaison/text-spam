import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\aswin\OneDrive\Desktop\spam\text-spam\results\model\spam_model')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokeniser = pickle.load(file)

# Function to preprocess and predict
def predict_spam(message):
    encoded_message = tokeniser.texts_to_sequences([message])
    padded_message = tf.keras.preprocessing.sequence.pad_sequences(encoded_message, maxlen=10, padding='post')
    prediction = (model.predict(padded_message) > 0.5).astype("int32")
    return prediction[0, 0]

# Streamlit app
def main():
    st.title("Spam Detection App")

    # Get user input
    user_input = st.text_input("Enter a message:")
    
    if st.button("Predict"):
        # Make prediction
        prediction = predict_spam(user_input)
        
        # Display result
        if prediction == 1:
            st.error("Spam!")
        else:
            st.success("Ham!")

    # Additional functionalities (optional)
    if st.checkbox("Show Dataset"):
        st.write("Showing dataset...")
        # Add code to display the dataset

    if st.checkbox("Show Model Summary"):
        st.write("Model Summary:")
        st.code(model.summary())

    if st.checkbox("Show Classification Report"):
        st.write("Classification Report:")
        # Add code to display classification report
        # ...

    if st.checkbox("Show Confusion Matrix"):
        st.write("Confusion Matrix:")
        # Add code to display confusion matrix
        # ...

    # You can add more interactive elements and visualizations based on your needs.

if __name__ == "__main__":
    main()
