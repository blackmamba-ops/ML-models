import speech_recognition as sr

def speech_to_text():
    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.adjust_for_ambient_noise(source)  # Optional: Adjust for ambient noise
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio)

        print("You said: ", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print(f"Error fetching results from Google Web Speech API: {e}")

if __name__ == "__main__":
    speech_to_text()


# When you run this code, it will listen to your speech through the microphone, and then attempt to convert it to text
# using Google Web Speech API. The recognized text will be printed to the console.