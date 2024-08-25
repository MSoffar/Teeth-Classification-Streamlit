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

# Function to simulate live typing effect on the same line
def stream_text(text):
    placeholder = st.empty()  # Create a placeholder
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(f"**AI Assistant:** {full_text}", unsafe_allow_html=True)
        time.sleep(0.01)  # Adjust the speed of the typing effect

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

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to the target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]

    # Display the prediction
    st.markdown(f"<h2>Your Tooth's Class is: {predicted_class} 😁</h2>", unsafe_allow_html=True)

    # Call GPT-4o-2024-08-06 for insights
    st.markdown("<h3>Getting insights...</h3>", unsafe_allow_html=True)

    # System prompt for GPT model
    system_prompt = (
        "You are a PhD dentist with great knowledge in dental care. "
        "You provide concise, accurate, and to-the-point responses. "
        "Focus on delivering clear and actionable advice."
    )

    # Example prompt to GPT-4o-2024-08-06
    prompt = f"You have classified a tooth as {predicted_class}. Provide detailed insights about this classification, including what it means, how it can be treated, and preventive measures."

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Using the specified model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    insights = response.choices[0].message.content.strip()

    # Display the insights with live typing effect
    stream_text(insights)

    # Chatbot interaction
    st.markdown("## Chat with Our AI-Powered Dental Assistant 🤖")

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                stream_text(message["content"])  # Display AI responses with typing effect

    # User input for chatbot
    user_input = st.text_input("Ask a question related to your tooth classification...")

    if user_input:
        # Add user's message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Example prompt to GPT-4o-2024-08-06 for chatbot
        chatbot_prompt = f"You have classified a tooth as {predicted_class}. The user asked: '{user_input}'. Provide a detailed response."

        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using the specified model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chatbot_prompt}
            ],
        )

        chatbot_response = response.choices[0].message.content.strip()

        # Add AI's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        # Display the response with typing effect
        stream_text(chatbot_response)

# Dental Health Quiz
st.markdown("## Dental Health Quiz 📝")

# Sample Quiz Questions with unique keys
st.write("How often do you brush your teeth?")
brush_freq = st.radio("", ('Once a day', 'Twice a day', 'More than twice'), key="brush_freq", index=None)

st.write("Do you floss regularly?")
floss_freq = st.radio("", ('Yes', 'No'), key="floss_freq", index=None)

st.write("Do you consume sugary foods or drinks often?")
sugar_consumption = st.radio("", ('Yes', 'No'), key="sugar_consumption", index=None)

# Processing only happens after clicking the button
if st.button("Get Recommendations"):
    # Provide recommendations based on answers
    recommendation_prompt = f"A person brushes their teeth {brush_freq}, flosses {floss_freq}, and consumes sugary foods: {sugar_consumption}. Provide personalized dental care recommendations."

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Using the specified model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": recommendation_prompt}
        ],
    )

    recommendations = response.choices[0].message.content.strip()

    st.markdown(f"**Personalized Recommendations:** {recommendations}")

# Gamification and Progress Tracking
st.markdown("## Track Your Progress & Earn Rewards 🎉")

# Example progress tracker
progress = st.progress(0)

if st.button("Complete Today's Task"):
    for i in range(1, 101):
        progress.progress(i)
        time.sleep(0.05)  # Simulate task completion
    st.success("Congratulations! You've earned a badge 🏅")

# Professional Consultation Integration
st.markdown("## Book a Professional Consultation 🦷")

st.write("If you're concerned about your classification, consider booking a consultation with a professional dentist.")

# Example booking button (could link to an external booking system)
if st.button("Book Now"):
    st.markdown("Call us on +20-1111111111", unsafe_allow_html=True)

# Footer with a call-to-action and emojis
st.markdown(
    "<footer style='text-align: center; font-size: 18px;'>"
    "Transform your dental care experience with AI-powered insights 🚀<br>",
    unsafe_allow_html=True
)
