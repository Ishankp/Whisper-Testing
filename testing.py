import whisper
import os

def transcribe_audio(audio_folder, output_folder):
    """Transcribes all .mp3 files in the audio_folder and saves the output in output_folder."""
    
    # Check if audio folder exists
    if not os.path.isdir(audio_folder):
        print(f"Error: Folder '{audio_folder}' not found.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the pre-trained model
    model = whisper.load_model("base")  # Choose from 'tiny', 'base', 'small', 'medium', or 'large'

    # Iterate through all .mp3 files in the folder
    for filename in os.listdir(audio_folder):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(audio_folder, filename)
            print(f"Transcribing: {filename}")

            # Transcribe the audio file (force FP32 to suppress warnings)
            result = model.transcribe(audio_path, fp16=False)

            # Get transcription text
            transcription = result["text"]
            print(f"Transcription for {filename}:\n{transcription}\n")

            # Define output file path in the output_folder
            output_file = os.path.join(output_folder, f"{filename}.txt")

            # Save the result to a text file in output_folder
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)

    print("Transcription process completed.")

def main():
    audio_folder = "audio_folder"
    output_folder = "output_folder"
    transcribe_audio(audio_folder, output_folder)

    

if __name__ == "__main__":
    main()
