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

# Function to simulate live typing effect
def simulate_typing(response_text, chat_placeholder, delay=0.03):
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

# Streamlit UI configuration
st.set_page_config(page_title="Teeth Classification and Chatbot", layout="centered")

# Custom CSS for background image and layout
st.markdown(
    f"""
    <style>
    .main {{
        background-image: url('dentist.jpg');
        background-size: cover;
        background-position: center;
        color: #E1E1E1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stTextInput {{
        background-color: #323232;
        color: #fff;
    }}
    .stButton button {{
        background-color: #0056b3;
        color: #fff;
        border: none;
        padding: 5px 15px;
        font-size: 14px;
        border-radius: 5px;
        margin: 5px 0;
    }}
    .stMarkdown {{
        color: #fff;
    }}
    .chat-history {{
        border-radius: 10px;
        padding: 20px;
        background-color: #2B2B2B;
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 10px;
    }}
    .logo {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .chat-input {{
        display: flex;
        align-items: center;
    }}
    .chat-input .stTextInput {{
        flex-grow: 1;
    }}
    .chat-input .stButton {{
        margin-left: -70px;
        z-index: 1;
    }}
    .chat-input .stButton button {{
        border-radius: 20px;
        padding: 2px 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo image centered
st.image("logo.jpg", width=150)  # Display your logo image

st.title("Teeth Classification with AI Magic 🦷✨")
st.subheader("Chat with Sally, your virtual assistant.")

# Placeholder for chat history
chat_placeholder = st.empty()

# Display initial chat history
chat_placeholder.markdown(assemble_chat(st.session_state.messages))

# Function to process user input and reset the input field
def process_input():
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Add user message to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Update chat history with the user's message
        chat_placeholder.markdown(assemble_chat(st.session_state.messages))

        # Get model response
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=st.session_state.messages
            )

            # Extract assistant message
            assistant_message = completion.choices[0].message.content.strip()

            # Simulate typing effect before adding to the history
            simulate_typing(assistant_message, chat_placeholder)

            # Append the assistant's message to the conversation history
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

            # Update chat history with the full message after typing simulation
            chat_placeholder.markdown(assemble_chat(st.session_state.messages))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Reset the input field
        st.session_state.user_input = ""

# File uploader for teeth classification
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
        chat_placeholder.markdown(assemble_chat(st.session_state.messages))

    # User input for chatbot
    st.text_input("Ask a question related to your tooth classification...", key="user_input", on_change=process_input)
