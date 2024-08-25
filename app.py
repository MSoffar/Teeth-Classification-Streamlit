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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate live typing effect
def simulate_typing(response_text, chat_placeholder, delay=0.005):
    typed_text = ""
    for char in response_text:
        typed_text += char
        chat_placeholder.markdown(assemble_chat(st.session_state.messages, typed_text))
        time.sleep(delay)

# Function to assemble chat history
def assemble_chat(messages, current_response=""):
    chat_history = ""
    for message in messages:
        if message["role"] == "user":
            chat_history += f"**You:** {message['content']}\n\n"
        elif message["role"] == "assistant":
            chat_history += f"**AI Assistant:** {message['content']}\n\n"
    if current_response:
        chat_history += f"**AI Assistant:** {current_response}"
    return chat_history

# OpenAI GPT-4o setup
openai.api_key = st.secrets["openai"]["api_key"]

# Placeholder for chat history
chat_placeholder = st.empty()

# Streamlit app title
st.image("logo.jpg", width=150)  # Display your logo image
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

    # Display initial chat history
    chat_placeholder.markdown(assemble_chat(st.session_state.messages))

    # User input for chatbot
    user_input = st.text_input("Ask a question related to your tooth classification...", key="user_input")

    # Display the submit button for chatbot input
    if st.button("Submit Query") and user_input:
        # Add user's message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Example prompt to GPT-4o-2024-08-06 for chatbot
        chatbot_prompt = f"You have classified a tooth as {st.session_state.predicted_class}. The user asked: '{user_input}'. Provide a detailed response."

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",  # Using the specified model
                messages=[
                    {"role": "system", "content": "You are a PhD dentist with great knowledge in dental care. Provide concise, accurate, and actionable advice."},
                    {"role": "user", "content": chatbot_prompt}
                ],
            )

            chatbot_response = response.choices[0].message.content.strip()

            # Add AI's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

            # Stream the response with typing effect
            simulate_typing(chatbot_response, chat_placeholder)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Reset the input field
        st.session_state.user_input = ""
