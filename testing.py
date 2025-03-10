import whisper
import torch
import os

# Define folder paths
audio_folder = "audio_folder"
output_folder = "output_folder"

# Check if audio folder exists
if not os.path.isdir(audio_folder):
    print(f"Error: Folder '{audio_folder}' not found.")
    exit(1)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained model
model = whisper.load_model("base")  # You can use 'tiny', 'base', 'small', 'medium', or 'large'

# Iterate through all .mp3 files in the folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, filename)
        print(f"Transcribing: {filename}")

        # Transcribe the audio file
        result = model.transcribe(audio_path, fp16=False)  # Force FP32 to avoid the warning

        # Print and save the transcription
        transcription = result["text"]
        print(f"Transcription for {filename}:\n{transcription}\n")

        # Define output file path in the output_folder
        output_file = os.path.join(output_folder, f"{filename}.txt")

        # Save the result to a text file in output_folder
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

print("Transcription process completed.")
