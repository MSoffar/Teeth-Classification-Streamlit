import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import tempfile
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

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "chatbot_insights" not in st.session_state:
    st.session_state.chatbot_insights = []  # Store chatbot responses


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
            chat_history += f"You: {message['content']}\n\n"
        elif message["role"] == "assistant":
            chat_history += f"AI Assistant: {message['content']}\n\n"
    if current_response:
        chat_history += f"AI Assistant: {current_response}"
    return chat_history


# OpenAI GPT-4o setup
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit app title
st.image("logo.jpg", width=150)  # Display your logo image
st.title("Teeth Classification with AI Magic 🦷✨")

# User personal information
st.markdown("### Please enter your details:")
name = st.text_input("Name:")
gender = st.selectbox("Gender:", ["Please choose", "Male", "Female", "Other"])
age = st.number_input("Age:", min_value=0, max_value=120, step=1)
email = st.text_input("Enter your email to receive the report:")

# Additional questions
st.markdown("### Dental Health Questions:")
brushing_frequency = st.selectbox("How often do you brush your teeth?",
                                  ["Please choose", "Once a day", "Twice a day", "More than twice", "Rarely"])
dentist_visits = st.selectbox("How often do you visit the dentist?",
                              ["Please choose", "Every 6 months", "Once a year", "Rarely", "Never"])
flossing_habits = st.selectbox("How often do you floss?",
                               ["Please choose", "Daily", "Weekly", "Rarely", "Never"])

# File uploader for image classification
st.markdown("### Upload an image to classify your tooth:")
uploaded_file = st.file_uploader("Choose an image to classify...", type="jpg")


# Function to validate inputs
def validate_inputs():
    if not name or age == 0 or brushing_frequency == "Please choose" or dentist_visits == "Please choose" or flossing_habits == "Please choose" or not uploaded_file or not email:
        return False
    return True


# Button to submit the image and make prediction
if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption='Uploaded Image 🖼️', use_column_width=False, width=300)

    if st.button("Submit Image"):
        if validate_inputs():
            # Preprocess the image
            img = Image.open(uploaded_file).resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = classes[np.argmax(predictions)]

            # Display the prediction
            st.write(f"### Your Tooth's Class is: {predicted_class} 😢")

            # Save the prediction in session state for use in the chatbot and report
            st.session_state.predicted_class = predicted_class
        else:
            st.error("Please fill in all required fields and upload an image before submitting.")

# Chatbot section
st.markdown("## Chat with Our AI-Powered Dental Assistant 🤖")

# Placeholder for chat history
chat_placeholder = st.empty()

# Display initial chat history
chat_placeholder.markdown(assemble_chat(st.session_state.messages))

# User input for chatbot
user_input = st.text_input("Ask a question related to your tooth classification...", key="user_input")

# Display the submit button for chatbot input
if st.button("Submit Query") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Example prompt to GPT-4o for chatbot, including the tooth classification
    chatbot_prompt = f"You have classified a tooth as {st.session_state.predicted_class}. The user asked: '{user_input}'. Provide a detailed response."

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Using the specified model
            messages=st.session_state.messages + [{"role": "user", "content": chatbot_prompt}],
        )

        chatbot_response = response.choices[0].message.content.strip()

        # Stream the response with typing effect
        simulate_typing(chatbot_response, chat_placeholder)

        # After streaming is finished, add the response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
        st.session_state.chatbot_insights.append(chatbot_response)  # Store the response for the final report

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Refresh the chat history to include the latest interaction
    chat_placeholder.markdown(assemble_chat(st.session_state.messages))


# Function to generate the personalized report using GPT-4o
def generate_personalized_report(predicted_class, user_details, chatbot_insights):
    prompt = f"""
    You are an AI dental assistant. Based on the user's details and the classification of their tooth condition as {predicted_class}, generate a structured, professional, and personalized report. Include a personal information section, an evaluation summary, immediate actions, long-term care, and professional consultation recommendations. Also, incorporate any insights provided by the chatbot in the previous conversation.

    User Details:
    - Name: {user_details['name']}
    - Gender: {user_details['gender']}
    - Age: {user_details['age']}
    - Brushing Frequency: {user_details['brushing_frequency']}
    - Dentist Visits: {user_details['dentist_visits']}
    - Flossing Habits: {user_details['flossing_habits']}

    Chatbot Insights: {chatbot_insights}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2  # Adjust temperature for more creativity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the report: {str(e)}")
        return None


# Function to send email with the report using Gmail and HTML formatting
def send_email_via_gmail(report_content, email, user_details, img_path=None):
    from_email = st.secrets["gmail"]["email"]
    password = st.secrets["gmail"]["password"]

    to_email = email
    subject = "Your Personalized Dental Health Report"

    # HTML content for the email
    html_content = f"""
    <html>
    <body>
        <h1 style='text-align: center; font-family: Arial, sans-serif; color: #4CAF50;'>Personalized Oral Health Report</h1>
        <h2 style="font-family: Arial, sans-serif;">Personal Information</h2>
        <p style="font-family: Arial, sans-serif; font-size: 14px;">
            <strong>Name:</strong> {user_details['name']}<br>
            <strong>Gender:</strong> {user_details['gender']}<br>
            <strong>Age:</strong> {user_details['age']}
        </p>
        <h2 style="font-family: Arial, sans-serif;">Evaluation Summary</h2>
        <p style="font-family: Arial, sans-serif; font-size: 14px;">Dear {user_details['name']},<br><br>{report_content}</p>
    </body>
    </html>
    """

    # Create the email message with HTML
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(html_content, 'html'))  # Attach HTML content

    # Attach the image if available
    if img_path:
        attachment = open(img_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {img_path}')
        msg.attach(part)

    try:
        # Connect to Gmail’s SMTP server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")


# Button to generate and send the personalized report via email
if st.button("Send Report via Email"):
    if validate_inputs():
        # Generate the personalized report
        user_details = {
            "name": name,
            "gender": gender,
            "age": age,
            "brushing_frequency": brushing_frequency,
            "dentist_visits": dentist_visits,
            "flossing_habits": flossing_habits
        }
        report_content = generate_personalized_report(st.session_state.predicted_class, user_details,
                                                      st.session_state.chatbot_insights)

        if report_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img_path = tmp_file.name
                tmp_file.write(uploaded_file.getvalue())

            send_email_via_gmail(report_content, email, user_details, img_path)
    else:
        st.error("Please fill in all required fields and upload an image before submitting.")
