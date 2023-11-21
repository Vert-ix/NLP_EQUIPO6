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


# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de la API de YouTube desde las variables de entorno
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Cargar el modelo XGBoost desde el archivo pickle
modelo_path = os.path.join(os.path.dirname(__file__), os.getenv("MODELO_PATH"))
with open(modelo_path, 'rb') as file:
    model = pickle.load(file)

# Cargar el vectoriza
vectorizer_path = os.path.join(os.path.dirname(__file__), os.getenv("VECTORIZER_PATH"))
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)  

stopword = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Función de preprocesamiento
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

# Obtener comentarios del video
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

# Redimensionar la imagen al 50% del tamaño original
new_size = (img.width // 5, img.height // 5)
resized_img = img.resize(new_size)

# Mostrar la imagen redimensionada
st.image(resized_img)

# Título de la aplicación
st.title("Detector de Comentarios Tóxicos")

# Entrada del ID del video
video_id_input = st.text_input('Ingresa el ID del video de YouTube:', key='video_id')

# Botones para obtener y limpiar comentarios en una sola fila
obtener_comentarios_btn, limpiar_comentarios_btn = st.columns(2)

# Espacio en blanco para mostrar comentarios
empty_space = st.empty()

# Verificar si se hizo clic en el botón "Obtener Comentarios"
if obtener_comentarios_btn.button("Obtener Comentarios", help="Haz clic para obtener comentarios del video"):
    # Verificar que se haya ingresado un ID de video
    if video_id_input:
        # Obtener comentarios del video
        comments = get_video_comments(youtube_api_key, video_id_input)

        # Analizar comentarios
        for comment in comments:
            # Preprocesar el comentario
            preprocessed_comment = preprocess_text(comment)

            # Vectorizar el comentario preprocesado
            comment_vectorized = vectorizer.transform([preprocessed_comment])

            # Realizar la predicción
            prediction = model.predict(comment_vectorized)

            # Mostrar el resultado
            if prediction[0] == 1:
                st.error(f"Comentario tóxico: {comment}")
            else:
                st.success(f"Comentario no tóxico: {comment}")

# Verificar si se hizo clic en el botón "Limpiar Comentarios"
if limpiar_comentarios_btn.button("Limpiar Comentarios", help="Haz clic para limpiar los comentarios mostrados"):
    empty_space.text("Comentarios limpiados.")


# Entrada de texto en la barra lateral para introducir un comentario
st.sidebar.title("Análisis de Comentarios")
user_comment = st.sidebar.text_area("Introduce un comentario para analizar:")
if st.sidebar.button("Analizar Comentario", help="Haz clic para analizar el comentario"):
    if user_comment:
        # Preprocesar el comentario
        preprocessed_user_comment = preprocess_text(user_comment)

        # Vectorizar el comentario preprocesado
        user_comment_vectorized = vectorizer.transform([preprocessed_user_comment])

        # Realizar la predicción
        user_comment_prediction = model.predict(user_comment_vectorized)

        # Mostrar el resultado
        if user_comment_prediction[0] == 1:
            st.sidebar.error("Este comentario es tóxico.")
        else:
            st.sidebar.success("Este comentario no es tóxico.")
    else:
        st.sidebar.warning("Por favor, ingrese un comentario antes de analizar.")