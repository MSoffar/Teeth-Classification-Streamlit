import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import openai
import time

# Load the trained model
model = load_model('teeth_classification_model.keras')  # Ensure this model path is correct

# Define class names
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Initialize OpenAI API with the secret key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Define system prompt for the GPT-4o chatbot
system_prompt = (
    "You are a PhD dentist with great knowledge in dental care. "
    "You provide concise, accurate, and to-the-point responses. "
    "Focus on delivering clear and actionable advice."
)

# Function to make predictions and display the result
def predict_and_display(image_file):
    try:
        # Load and preprocess the image
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make a prediction (batch size of 1 for stability)
        prediction = model.predict(img_array, batch_size=1)
        predicted_class = class_names[np.argmax(prediction)]
        
        # Display results
        st.write(f"### The detected dental disease is: **{predicted_class}** 😢")
        st.markdown("### 😱 😟 😔 😩 😖 😫 😞")
        st.image(image_file, caption=f'Predicted: {predicted_class}', use_column_width=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Function for chatbot interaction
def get_chatbot_response(chatbot_prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Using the specified model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chatbot_prompt}
            ]
        )
        chatbot_response =  response.choices[0].message.content.strip()
        return chatbot_response
    
    except Exception as e:
        st.error(f"An error occurred while getting the chatbot response: {str(e)}")
        return ""

# Streamlit app UI
st.title("Teeth Disease Classifier and Dental Care Chatbot 🦷")
st.write("Upload an image of teeth, and the model will classify the disease.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    predict_and_display(uploaded_file)

# Chatbot section
st.write("### Need dental advice? Ask our expert chatbot below:")
user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("The dentist is thinking..."):
        response = get_chatbot_response(user_input)
    
    # Create an empty container for live text display
    live_response = st.empty()

    # Simulate live typing by displaying one character at a time
    for i in range(len(response)):
        live_response.markdown(response[:i+1])
        time.sleep(0.03)  # Adjust typing speed by changing the sleep duration
