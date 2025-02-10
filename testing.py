import whisper

# Load the pre-trained model
model = whisper.load_model("base")  # You can use 'tiny', 'base', 'small', 'medium', or 'large'

# Load and transcribe the audio file
result = model.transcribe("audio.mp3")

# Print the transcription
print(result['text'])
