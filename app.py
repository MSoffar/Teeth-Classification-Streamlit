import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import openai
import time

# Load the classification model
model = load_model('teeth_classification_model.keras')

# Define the classes
classes = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Initialize session state for chat history and user input
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Function to simulate live typing effect on the same line
def stream_text(text):
    placeholder = st.empty()  # Create a placeholder
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(f"**AI Assistant:** {full_text}", unsafe_allow_html=True)
        time.sleep(0.005)  # Adjust the speed of the typing effect

# OpenAI GPT-4o setup
openai.api_key = st.secrets["openai_api_key"]

# Streamlit app title
st.title("Teeth Classification with AI Magic 🦷✨")

# File uploader
uploaded_file = st.file_uploader("Choose an image to classify...", type="jpg")

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption='Uploaded Image 🖼️', use_column_width=True)

    # Button to submit the image and make prediction
    if st.button("Submit Image"):
        # Preprocess the image
        img = Image.open(uploaded_file).resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]

        # Display the prediction
        st.write(f"### Your Tooth's Class is: {predicted_class} 😁")

        # Save the prediction in session state for use in the chatbot
        st.session_state.predicted_class = predicted_class

# Chatbot section
if "predicted_class" in st.session_state:
    st.markdown("## Chat with Our AI-Powered Dental Assistant 🤖")

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")

    # User input for chatbot
    user_input = st.text_input("Ask a question related to your tooth classification...", value=st.session_state.user_input, key="user_input")

    # Show the "Submit Query" button and process the query when clicked
    if st.button("Submit Query") and st.session_state.user_input:
        # Add user's message to chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})

        # Example prompt to GPT-4o-2024-08-06 for chatbot
        chatbot_prompt = f"You have classified a tooth as {st.session_state.predicted_class}. The user asked: '{st.session_state.user_input}'. Provide a detailed response."

        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Using the specified model
            messages=[
                {"role": "system", "content": "You are a PhD dentist with great knowledge in dental care. Provide concise, accurate, and actionable advice. Answer your response according to the part of {The user asked in the prompt}"},
                {"role": "user", "content": chatbot_prompt}
            ],
        )

        chatbot_response = response.choices[0].message.content.strip()

        # Add AI's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        # Stream the response with typing effect
        stream_text(chatbot_response)

        # Reset the user input field
        st.session_state.user_input = ""
