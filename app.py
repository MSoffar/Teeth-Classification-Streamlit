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

# Initialize session state for chat history, and submission flags
if "messages" not in st.session_state:
    st.session_state.messages = []
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "image_submitted" not in st.session_state:
    st.session_state.image_submitted = False  # Flag to track image submission

# OpenAI GPT-4o setup
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit app title
st.title("Teeth Classification with AI Magic 🦷✨")

# File uploader
uploaded_file = st.file_uploader("Choose an image to classify...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image 🖼️', use_column_width=True)

    if st.button("Submit Image"):  # Submit button for image classification
        # Preprocess the image
        img = img.resize((224, 224))  # Resize to the target size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]

        # Store the prediction in session state
        st.session_state.predicted_class = predicted_class
        st.session_state.image_submitted = True  # Update the flag to indicate submission

if st.session_state.image_submitted and st.session_state.predicted_class:
    # Display the prediction with a sad face
    st.markdown(f"<h2>Your Tooth's Class is: {st.session_state.predicted_class} 😟</h2>", unsafe_allow_html=True)

# Chatbot interaction
if st.session_state.image_submitted and st.session_state.predicted_class:
    st.markdown("## Chat with Our AI-Powered Dental Assistant 🤖")

    # Display chat history without streaming
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")

    # User input for chatbot
    user_input = st.text_input("Ask a question related to your tooth classification...")

    if st.button("Submit Question"):  # Submit button for chatbot
        if user_input:
            # Add user's message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # System prompt for GPT model
            system_prompt = (
                "You are a PhD dentist with great knowledge in dental care. "
                "You provide concise, accurate, and to-the-point responses. "
                "Focus on delivering clear and actionable advice."
            )

            # Example prompt to GPT-4o-2024-08-06 for chatbot
            chatbot_prompt = f"You have classified a tooth as {st.session_state.predicted_class}. The user asked: '{user_input}'. Provide a detailed response."

            # Stream GPT response
            response = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",  # Using the specified model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chatbot_prompt}
                ],
                stream=True  # Stream the response
            )

            # Placeholder for streaming text
            full_response = ""
            for chunk in response:
                chunk_text = chunk['choices'][0].get('delta', {}).get('content', '')
                full_response += chunk_text
                st.write(f"**AI Assistant:** {full_response}")

            # Add AI's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Clear the input field after submission
            st.experimental_rerun()
# Footer with a call-to-action and emojis
st.markdown(
    "<footer style='text-align: center; font-size: 18px;'>"
    "Transform your dental care experience with AI-powered insights 🚀<br>",
    unsafe_allow_html=True
)
