import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Retrieve the YouTube API key from environment variables
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Load XGBoost model from pickle file
modelo_path = os.path.join(os.path.dirname(__file__), os.getenv("MODELO_PATH"))
with open(modelo_path, 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
vectorizer_path = os.path.join(os.path.dirname(__file__), os.getenv("VECTORIZER_PATH"))
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)  

stopword = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove non-alphabetic characters 
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove stopwords
    text = text.split()
    text = [word for word in text if word not in stopword]

    # Lemmatize
    text = [lemmatizer.lemmatize(word) for word in text]

    # Tokenize the text
    tokens = [token.lower() for token in word_tokenize(' '.join(text))]   
    
    # Join the tokens into a single string
    text = ' '.join(tokens)

    return text

# Get comments from the video
def get_video_comments(youtube_api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText'
    )
    comments = []
    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments


image_path = 'img/youtube.jpg'  
img = Image.open(image_path)

# Resize the image to 50% of the original size
new_size = (img.width // 5, img.height // 5)
resized_img = img.resize(new_size)

# Display the resized image
st.image(resized_img)

# App Title
st.title("Detector de Comentarios Tóxicos")

# Video ID Input
video_id_input = st.text_input('Ingresa el ID del video de YouTube:', key='video_id')

# Buttons to fetch and clear comments in a single row
obtener_comentarios_btn, limpiar_comentarios_btn = st.columns(2)

# Space to display comments
empty_space = st.empty()

# Check if the 'Get Comments' button was clicked
if obtener_comentarios_btn.button("Obtener Comentarios", help="Haz clic para obtener comentarios del video"):
    # Check that a video ID has been entered
    if video_id_input:
        # Get video comments
        comments = get_video_comments(youtube_api_key, video_id_input)

        # Analyze comments
        for comment in comments:
            # Preprocess coments
            preprocessed_comment = preprocess_text(comment)

            # Vectorize
            comment_vectorized = vectorizer.transform([preprocessed_comment])

            # Predict
            prediction = model.predict(comment_vectorized)

            # Result
            if prediction[0] == 1:
                st.error(f"Comentario tóxico: {comment}")
            else:
                st.success(f"Comentario no tóxico: {comment}")

# Check if the 'Clean Comments' button was clicked
if limpiar_comentarios_btn.button("Limpiar Comentarios", help="Haz clic para limpiar los comentarios mostrados"):
    empty_space.text("Comentarios limpiados.")


# Text input in the sidebar to enter a comment
st.sidebar.title("Análisis de Comentarios")
user_comment = st.sidebar.text_area("Introduce un comentario para analizar:")
if st.sidebar.button("Analizar Comentario", help="Haz clic para analizar el comentario"):
    if user_comment:
        # Preprocess
        preprocessed_user_comment = preprocess_text(user_comment)

        # Vectorize
        user_comment_vectorized = vectorizer.transform([preprocessed_user_comment])

        # Predict
        user_comment_prediction = model.predict(user_comment_vectorized)

        # Result
        if user_comment_prediction[0] == 1:
            st.sidebar.error("Este comentario es tóxico.")
        else:
            st.sidebar.success("Este comentario no es tóxico.")
    else:
        st.sidebar.warning("Por favor, ingrese un comentario antes de analizar.")