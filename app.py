import cv2
import numpy as np
import joblib
import mediapipe as mp
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import pickle
import streamlit as st

# Function to convert sign to text
# Load the trained model
with open("D:\MY FILES\My Coding Projects\Sign language Project\model1.pickle", 'rb') as f:
    model = pickle.load(f)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,  max_num_hands=2, min_detection_confidence=0.3)

def recognize_gesture(frame, model):
    """Recognizes sign language gestures from a webcam frame."""
    data_aux = []
    x_ = []
    y_ = []

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks using MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks and normalize coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize coordinates by shifting to the minimum x and y
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure the feature length matches the trained model's expectations
            if len(data_aux) == model.n_features_in_:
                # Predict the sign word using the trained model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_word = prediction[0]

                # Display predicted word on frame
                cv2.putText(frame, predicted_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                return predicted_word  # Return the recognized word

    return None  # No gesture detected
    
import pyttsx3    
tts_engine = pyttsx3.init()

# Function to convert speech to sign videos
# Path to the directory containing gesture videos
gesture_video_path = "D:\MY FILES\My Coding Projects\Sign language Project\Sign videos"

# Mapping recognized words to gesture videos
gesture_mapping = {
    "Bad": f"{gesture_video_path}/Bad.mp4",
    "thank you": f"{gesture_video_path}/Thank you.mp4",
    "welcome": f"{gesture_video_path}/Welcome.mp4",
    "how are you": f"{gesture_video_path}/How are you.mp4",
    "beautiful": f"{gesture_video_path}/Beautiful.mp4",
    "what is your name": f"{gesture_video_path}/What is your name.mp4",
    "teach": f"{gesture_video_path}/Teach.mp4",
    "teacher": f"{gesture_video_path}/Teacher.mp4",
    "today": f"{gesture_video_path}/Today.mp4",
    "tomorrow": f"{gesture_video_path}/Tomorrow.mp4",
    "travel": f"{gesture_video_path}/Travel.mp4",
    "university": f"{gesture_video_path}/University.mp4",
    "what is your occupation": f"{gesture_video_path}/What is your occupation.mp4",
    "woman": f"{gesture_video_path}/Woman.mp4",
    "work": f"{gesture_video_path}/Work.mp4",
    "yes": f"{gesture_video_path}/Yes.mp4",
    "yesterday": f"{gesture_video_path}/Yesterday.mp4",
    "marriage": f"{gesture_video_path}/Marriage.mp4",
    "mother": f"{gesture_video_path}/Mother.mp4",
    "my name is Ezra": f"{gesture_video_path}/My name is Ezra.mp4",
    "no": f"{gesture_video_path}/No.mp4",
    "peace": f"{gesture_video_path}/Peace.mp4",
    "house": f"{gesture_video_path}/House.mp4",
    "how old are you": f"{gesture_video_path}/How old are you.mp4",
    "man": f"{gesture_video_path}/Man.mp4",
    "go": f"{gesture_video_path}/Go.mp4",
    "good afternoon": f"{gesture_video_path}/Good afternoon.mp4",
    "good evening": f"{gesture_video_path}/Good evening.mp4",
    "good morning": f"{gesture_video_path}/Good morning.mp4",
    "good": f"{gesture_video_path}/Good.mp4",
    "divorce": f"{gesture_video_path}/Divorce.mp4",
    "family": f"{gesture_video_path}/Family.mp4",
    "father": f"{gesture_video_path}/Father.mp4",
    "fine": f"{gesture_video_path}/Fine.mp4",
    "friend": f"{gesture_video_path}/Friend.mp4",
    "brother": f"{gesture_video_path}/Brother.mp4",
    "come": f"{gesture_video_path}/Come.mp4",
    "country": f"{gesture_video_path}/Country.mp4"
}

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
    cap.release()
    
# Function to convert text to speech
def text_to_speech(text):
    tts_engine.say(text)  # Convert text to speech
    tts_engine.runAndWait()  # Wait for the speech to finish

# Function to convert speech to text
def record_audio(filename="output.wav", duration=5, samplerate=44100):
    st.write("Recording audio... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, samplerate, audio)
    return filename

def recognize_speech_from_file(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        st.write("Processing audio for speech recognition...")
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data).lower()
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition; {e}")
    return None

# Set Streamlit page configuration

# Display centered logo at the top
st.image("D:\MY FILES\My Coding Projects\Sign language Project\sign_language_logo.png",width=150)

# Sidebar info
st.sidebar.info("This AI-powered system enables Bidirectional Communication between Hearing, Non-Hearing and Visually-Impaired individuals through intelligent multimodal translation of speech, text and sign language.")

def main():
    st.markdown(
        """
        <h1 style='text-align: left; color: blue;'>SignLink</h1>
        <h4 style='text-align: left; color: blue;'>A Multimodal Bidirectional Sign Language Translator</h4>
        """,
        unsafe_allow_html=True
    )

    mode = st.sidebar.selectbox("Choose a Mode", ["Speech-to-Text", "Speech-to-Sign", "Text-to-Sign", "Gesture Recognition"])

    if mode == "Speech-to-Text":
        st.header("🎙️Speech-to-Text")
        if st.button("Start Recording"):
            audio_filename = record_audio()
            recognized_text = recognize_speech_from_file(audio_filename)
            if recognized_text:
                st.write(f"Recognized Text: {recognized_text}")

    elif mode == "Speech-to-Sign":
        st.header("🎤Speech-to-Sign")
        if st.button("Start Recording"):
            audio_filename = record_audio()
            command = recognize_speech_from_file(audio_filename)
            if command in gesture_mapping:
                st.write(f"Playing video for: {command}")
                play_video(gesture_mapping[command])
            else:
                st.error("No matching sign found for spoken word.")

    elif mode == "Text-to-Sign":
        st.header("📜Text-to-Sign")
        user_input = st.text_input("Enter text to translate to sign")

        if user_input:  # Ensure input is provided before showing the button
            if st.button("Translate"):
                if user_input in gesture_mapping:
                    st.write(f"Playing video for: {user_input}")
                    play_video(gesture_mapping[user_input])
                else:
                    st.error("No matching sign found for the entered text.")



    elif mode == "Gesture Recognition":
        st.header("🤟Gesture Recognition")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        prediction_placeholder = st.empty()
        if st.button("Start Webcam"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = recognize_gesture(frame, model)
                stframe.image(frame, channels="RGB")
                prediction_placeholder.write(f"Predicted Gesture: {prediction}")
            cap.release()
    st.markdown("""
    © 2025 **SignLink: A Multimodal Bidirectional Sign Language Translator**  
    Developed by **Ezra Yalley**  
    📧 Contact: [ezrayalley@example.com](mailto:ezrayalley@example.com)  
    📞 Phone: +233 248 449 339
    """)

if __name__ == "__main__":
    main()