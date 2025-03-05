import whisper
import torch
import os

# Look for audio folder
audio_folder = "audio_folder"

# Check if folder exists
if not os.path.isdir(audio_folder):
    print(f"Error: Folder '{audio_folder}' not found.")
    exit(1)

# Load the pre-trained model
model = whisper.load_model("base")  # You can use 'tiny', 'base', 'small', 'medium', or 'large'


# Iterate through all .mp3 files in the folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, filename)
        print(f"Transcribing: {filename}")

        # Transcribe the audio file
        result = model.transcribe(audio_path)

        # Print and save the transcription
        transcription = result["text"]
        print(f"Transcription for {filename}:\n{transcription}\n")

        # Save the result to a text file
        output_file = os.path.join(audio_folder, f"{filename}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

print("Transcription process completed.")