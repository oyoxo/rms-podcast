import openai
import time
from rich import print
import requests
import pandas as pd 
import streamlit as st
from newspaper import Article
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from elevenlabs import generate, save, VoiceSettings, Voice
from pydub import AudioSegment
import os
from io import BytesIO
from elevenlabs.api import User
# Page configuration - #https://symbl.cc/en/emoji/animals-and-nature/
st.set_page_config(
    page_title="Generate Netgen AI Podcast",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Define a callback function that updates the session state
def on_change():
    st.session_state.news_df = news_df
    
# Initialize default values if not present in session state
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'text_generated' not in st.session_state:
    st.session_state.text_generated = False
if 'word_range' not in st.session_state:
    st.session_state.word_range = (750, 850)
if 'duration' not in st.session_state:
    st.session_state.duration = 5  # in minutes
if 'intro_text' not in st.session_state:
    st.session_state.intro_text = f"""Bienvenue dans le résumé de l'édition actuelle de la REVUE MÉDICALE SUISSE, numéro 855 intitulée 'Médecine Palliative'. Découvrez les dernières avancées dans le domaine médical."""
if 'ending_text' not in st.session_state:
    st.session_state.ending_text = f"""Pour approfondir votre compréhension des sujets abordés dans cette édition, visitez revmed.ch. Restez informé avec la REVUE MÉDICALE SUISSE."""
if 'podcast' not in st.session_state:
    st.session_state.podcast = ""
if 'prompt' not in st.session_state:    
    st.session_state.prompt = ""
if 'lower_podcast' not in st.session_state:
    st.session_state.lower_podcast = 5
if 'lower_background' not in st.session_state:
    st.session_state.lower_background = 25
# Initialize your DataFrame here as needed
if 'news_df' not in st.session_state or st.session_state.news_df.empty:
    st.session_state.news_df = pd.DataFrame(
        columns=['User', 'Rank','Title', 'URL', 'Text']
    )
if 'filtered_news_df' not in st.session_state:
    st.session_state.filtered_news_df = pd.DataFrame()
    
# API keys setup
openai.api_key = st.secrets["openai"]
eleven_api_key = st.secrets["elevenlabs"]

def setup_gspread():
    # Define the scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    # Setup the credentials
    credentials = ServiceAccountCredentials.from_json_keyfile_dict({
        "type": "service_account",
        "project_id": st.secrets["gs-project_id"],
        "private_key_id": st.secrets["gs-private_key_id"],
        "private_key": st.secrets["gs-private_key"],
        "client_email": st.secrets["gs-client_email"],
        "client_id": st.secrets["gs-client_id"],
        "auth_uri": st.secrets["gs-auth_uri"],
        "token_uri": st.secrets["gs-token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gs-auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gs-client_x509_cert_url"]
    }, scope)

    # Authorize and return client
    return gspread.authorize(credentials)

# Streamlit UI layout
col1, col2, col3 = st.columns(3) #all col equal st.columns([2, 5, 3]) diffenent col widths

# Sidebar for settings
with st.sidebar:
    st.title("Podcast Settings")
    #input_choice = st.radio("Select Input Method", ('URLs', 'Manual Input'))
    st.session_state.word_range = st.slider("Word Range", 100, 1000, st.session_state.word_range, step=5)
    #st.session_state.duration = st.slider("Duration (minutes)", 1, 10, st.session_state.duration)
    st.session_state.intro_text = st.text_area("Podcast Intro Text", st.session_state.intro_text, height=200)
    st.session_state.ending_text = st.text_area("Podcast Ending Text", st.session_state.ending_text, height=200)
    st.session_state.lower_podcast = st.slider("Podcast Volume Decrease)", 5, 50, st.session_state.lower_podcast, step=5)
    st.session_state.lower_background = st.slider("Background Music Volume Decrease)", 25, 50, st.session_state.lower_background, step=5)

def read_sheet_to_dataframe(worksheet):
    # Get all values from the worksheet
    data = worksheet.get_all_values()
    
    # Create a DataFrame with the data
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # Convert the 'Rank' column to numeric (int or float) to sort
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')  # Use 'coerce' to handle non-numeric values
    
    # Sort the DataFrame by the 'Rank' column, ascending
    df.sort_values(by='Rank', inplace=True, ascending=True)
    
    # Return the DataFrame
    return df

# Functions for audio processing 
# check https://github.com/JoelKronander/ttsDev https://github.com/JoelKronander/ttsDev/blob/main/ttsDev/main.py
def download_audio(url):
    response = requests.get(url, stream=True)  # Use stream=True to handle large files
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download audio: Status code {response.status_code}")

def mix_audio_podcast(podcast_audio_file, background_music_path, output_file):
    podcast = AudioSegment.from_file(podcast_audio_file, format="mp3")
    background_music = AudioSegment.from_file(background_music_path, format="mp3")

    # Adjust volumes, mix, and export
    podcast = podcast - int(st.session_state.lower_podcast)  # Decrease podcast volume 5
    background_music = background_music - int(st.session_state.lower_background)  # Decrease music volume 25

    if len(background_music) < len(podcast):
        background_music = background_music * (len(podcast) // len(background_music) + 1)

    mixed = podcast.overlay(background_music)
    mixed.export(output_file, format="mp3")

# Function to generate text using OpenAI
def generate_text(prompt, model="gpt-4-1106-preview", temperature=0.4, max_retries=3):
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message["content"].strip()
        except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
            print(f"Caught an error: {e}. Retrying in 60 seconds...")
            time.sleep(60)
    raise Exception("OpenAI API is unavailable after several retries.")

# Function to process URL with Newspaper3k
def process_direct_url(url):
    if url.startswith('http://') or url.startswith('https://'):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return f"An error occurred while processing the URL: {e}\n"
    else:
        return "URL provided is invalid or empty.\n"

# Function to process multiple URLs
def process_urls(urls):
    concatenated_text = ""
    for index, url in enumerate(urls, start=1):
        if url.strip():  # Check if URL is not empty
            article_text = process_direct_url(url)
            if article_text:
                concatenated_text += f"\n\n--- Article {index} ---\n\n{article_text}"
            else:
                concatenated_text += f"\n\n--- Article {index} ---\n\nError processing this URL."
        else:
            concatenated_text += f"\n\n--- Article {index} ---\n\nNo URL provided."
    return concatenated_text

# Function to get the user information from ElevenLabs
def get_elevenlabs_user_info(api_key):
    url = "https://api.elevenlabs.io/v1/user"
    headers = {"xi-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching ElevenLabs user info: {response.text}")

# Function to generate audio with ElevenLabs
def generate_audio(text, voice, filename):
    #settings=VoiceSettings(speaking_rate=1.1, stability=0.25, similarity_boost=0.75, style=0.0, use_speaker_boost=True) 
    settings = VoiceSettings(stability=0.5, similarity_boost=0.75,style=0.0, use_speaker_boost=True)# Adjust the speaking rate here
    audio = generate(text=text, voice=voice, api_key=eleven_api_key, model='eleven_multilingual_v2')
    save(audio, filename)
    return filename

def count_words(text):
    # Split the text into words based on spaces
    words = text.split()
    # Count the number of words
    return len(words)

def split_text_into_chunks(text, chunk_size=750):
    words = text.split()
    chunks = []
    current_chunk = words[0]
    for word in words[1:]:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    chunks.append(current_chunk)  # Add the last chunk
    return chunks

def generate_audio_for_chunks(chunks, voice):
    filenames = []
    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.mp3"
        generate_audio(chunk, voice, filename)
        filenames.append(filename)
    return filenames

def stitch_audio_files(filenames, output_filename):
    combined = AudioSegment.empty()
    for filename in filenames:
        segment = AudioSegment.from_mp3(filename)
        combined += segment
    combined.export(output_filename, format="mp3")

# Streamlit UI for column 1
# Streamlit UI for column 1
with col1:
    st.title("Generate Podcast Content")
    correct_code = "netgenai3k1"
    user_code = st.text_input("Enter the access code to proceed:", type="password")

    if user_code == correct_code:
        # Display the news dataframe
        #st.write(st.session_state.news_df)
        # Read the data into a DataFrame
        # Initialize the Google Sheet and worksheet
        # Setup gspread client
        client = setup_gspread()
        spreadsheet = client.open("Revmed_AI_Podcast")  # Replace with your actual sheet name
        worksheet = spreadsheet.worksheet("Sheet1")  # Replace with your actual worksheet name

        st.session_state.news_df = read_sheet_to_dataframe(worksheet)
        #print(st.session_state.news_df)
        #st.dataframe(st.session_state.news_df, use_container_width=True)
    
        edited_df = st.data_editor(st.session_state.news_df, num_rows="dynamic")

        # Update the session state with the edited DataFrame
        st.session_state.news_df = edited_df

        st.write("Enter URLs, each on a new line, to concatenate their texts.")
        urls_input = st.text_area("Enter URLs here:", key='url_input')

        if st.button("Process Content"):
            concatenated_text = ''

            # Process URLs if any are provided
            if urls_input.strip():
                urls = urls_input.split('\n')
                concatenated_text += process_urls(urls)
                st.write(f"Processed URL text: {concatenated_text}")

            # Get the manual text from the data editor
            if not st.session_state.news_df.empty:
                concatenated_manual_text = '\n\n'.join(st.session_state.news_df['Text'].tolist())
                concatenated_text += '\n\n' + concatenated_manual_text
                st.write(f"Processed text: {concatenated_manual_text}")

            if concatenated_text.strip():
                # Generate the podcast text
                word_range = st.session_state.word_range
                duration = st.session_state.duration
                intro_text = st.session_state.intro_text
                ending_text = st.session_state.ending_text
                st.session_state.prompt = f"""Erstelle einen Podcast-Text in Französich anhand der gegebenen Artikel, der eine Zusammenfassung der aktuellen Ausgabe der Zeitschrift REVUE MÉDICALE SUISSE darstellt.
                Die Zusammenfassung sollte die wichtigsten Forschungsergebnisse oder Themen beinhalten, die in der Ausgabe behandelt werden. 
                Achten Sie darauf, die Informationen präzise und detailliert, jedoch so kompakt wie möglich zu halten.
                Stellen Sie sicher, dass die Zusammenfassung die Neugier der Zuhörer weckt und sie dazu anregt, die vollständige Ausgabe zu erwerben oder ein Abonnement abzuschließen.
                Die Informationen sollten aktuell und von hoher Relevanz für die Zielgruppe sein.
                Die Länge des Textes soll ca. {word_range[0]} bis {word_range[1]} Wörter sein.
                Der Podcast beginnt mit {intro_text} und endet mit {ending_text}.
                Hier sind die Artikel Texte: {concatenated_text}"""
                st.session_state.generated_text = generate_text(st.session_state.prompt)
                st.session_state.text_generated = True
                st.write("Podcast text generated.")
            else:
                st.write("No content to process.")
    else:
        st.warning("Please enter a valid code to proceed.")

                    
# Streamlit UI for column 2 - Display the generated text if available
with col2:
    if user_code == correct_code and st.session_state.text_generated:
        st.title("Generated Podcast Text:")
        char_count = len(st.session_state.generated_text)
        #st.write(f"Number of characters in the generated podcast text: {char_count}")
        word_count = count_words(st.session_state.generated_text)
        st.write(f"Number of characters in the generated podcast text: {char_count} Number of words: {word_count}")
        #st.text_area("Edit generated podcast text", value=st.session_state.generated_text, height=600, key='generated_text')
        # Use a key for the text_area and a callback to update the session state
        edited_text = st.text_area("Edit generated podcast text", value=st.session_state.generated_text, height=600, key='generated_text_area')
        st.session_state.generated_text = edited_text  # Update session state when text changes

# Streamlit UI for column 3 - Select voice and generate podcast audio
with col3:
    if user_code == correct_code and st.session_state.text_generated:
        # Get user information
        user_info = get_elevenlabs_user_info(eleven_api_key)

        # Extract subscription details
        subscription = user_info['subscription']
        character_count = subscription['character_count']
        character_limit = subscription['character_limit']
        remaining_characters = character_limit - character_count
        # Format the number with thousands separator without setting a locale
        formatted_remaining_characters = f"{remaining_characters:,}".replace(",", ".")

    
        st.title("Generate Podcast Audio:")
        st.write(f"Remaining characters in ElevenLabs subscription: {formatted_remaining_characters}")
        options = ["Bill","George", "Antoni", "Grace", "Bella", "Dorothy", "Elli", "Rachel", "Adam", "Domi", "Josh", "Sam"]
        voice = st.selectbox("Select a voice for the podcast:", options)

        if st.button("Generate Podcast Audio"):
            with st.spinner('Generating audio Podcast...'):
                podcast_text = st.session_state.generated_text
                if podcast_text:
                    text_chunks = split_text_into_chunks(podcast_text)
                    chunk_files = generate_audio_for_chunks(text_chunks, voice)
                    # Stitching the audio files
                    podcast_audio_file = "full_podcast_audio.mp3"
                    stitch_audio_files(chunk_files, podcast_audio_file)
                    #podcast_audio_file = "podcast_audio.mp3"
                    #generate_audio(podcast_text, voice, podcast_audio_file)
                    st.audio(podcast_audio_file, format='audio/mp3')
                    
                    st.write("Now adding some background music and mastering the podcast...")
                    mixed_audio_file = "mixed_podcast.mp3"
                    mix_audio_podcast(podcast_audio_file, os.path.join("assets", "background_music.mp3"), mixed_audio_file)

                    st.audio(mixed_audio_file, format='audio/mp3')

# After defining all columns, ensure that the 'generated_text' is not overwritten after the widget is created
if 'Save Changes' in st.session_state:
    st.session_state.generated_text = st.session_state.generated_text
