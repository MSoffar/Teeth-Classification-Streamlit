# Teeth Classification with AI Magic 🦷✨

Welcome to the **Teeth Classification with AI Magic** project! This project uses a TensorFlow-based deep learning model to classify teeth images and provides insights using OpenAI's GPT-4o model. The application is built using Streamlit, making it easy to deploy and use as a web application.

## Features

- **Teeth Classification:** Upload an image of a tooth, and the model will classify it into one of seven categories: `CaS`, `CoS`, `Gum`, `MC`, `OC`, `OLP`, `OT`.
- **AI-Powered Insights:** The application uses the `gpt-4o-2024-08-06` model from OpenAI to provide concise, accurate, and to-the-point insights about the classified tooth.
- **Dental Health Quiz:** A quiz to assess your dental habits, followed by personalized recommendations.
- **Gamification:** Track your progress and earn badges for completing daily tasks.
- **Professional Consultation:** A feature to help users book a consultation with a dentist.

## Project Structure

```
├── app.py                   # Main application code
├── teeth_classification_model.keras  # Trained deep learning model
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

#Installation
To run this project locally, follow these steps:

Clone the repository:
git clone https://github.com/yourusername/teeth-classification-ai.git
cd teeth-classification-ai
Install the required dependencies:
Ensure you have Python 3.7+ installed.
pip install -r requirements.txt
Download the model:
Place the teeth_classification_model.keras file in the root directory of the project.
Set up OpenAI API Key:
You'll need an OpenAI API key to access GPT-4o. Set up your key in the app.py file:
openai.api_key = "your_openai_api_key"
Usage:
Run the application:
streamlit run app.py
Upload a Tooth Image:

Once the application is running, you can upload an image of a tooth. The model will classify the image and provide insights using GPT-4o.

Take the Dental Health Quiz:

Answer a few questions about your dental habits, and receive personalized recommendations based on your answers.

Track Progress and Earn Rewards:

Use the progress tracking feature to monitor your dental care routine and earn badges.

Book a Professional Consultation:

If you're concerned about the classification, you can use the consultation feature to book a visit with a professional dentist.

Deployment
You can deploy this application on Streamlit Cloud or any other hosting platform that supports Python and Streamlit.


Acknowledgments
TensorFlow: For providing the deep learning framework.
OpenAI: For the powerful GPT-4o model.
Streamlit: For making web application development easy and accessible.
