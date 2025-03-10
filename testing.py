import whisper
import os
import librosa
import Levenshtein

# Folders
audio_folder = "audio_folder"
output_folder = "output_folder"
report_folder = "report"

# Ensure output folder and report folder exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

# Function to calculate WPE (Word Per Error Rate)
def calculate_wpe(transcription, ground_truth):
    # Calculate Levenshtein distance (edit distance) between transcription and ground truth
    distance = Levenshtein.distance(transcription.lower(), ground_truth.lower())  # Case-insensitive
    wpe = distance / len(ground_truth.split()) if len(ground_truth.split()) > 0 else 0
    return wpe

# Function to calculate WPM (Words Per Minute)
def calculate_wpm(transcription, audio_path):
    # Load audio to get duration
    duration = librosa.get_duration(path=audio_path)  # Get duration in seconds
    duration_minutes = duration / 60  # Convert to minutes
    
    # Count words in the transcription
    word_count = len(transcription.split())
    
    # Calculate WPM
    wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    return wpm, duration, word_count

# Function to process each audio file
def process_audio_files():
    # Load the pre-trained Whisper model
    model = whisper.load_model("base")
    
    # Files to store results
    wpe_file = os.path.join(report_folder, "word_per_error_rate.txt")
    wpm_file = os.path.join(report_folder, "word_per_minute_rate.txt")
    
    # Open files to write results
    with open(wpe_file, "w", encoding="utf-8") as wpe_out, open(wpm_file, "w", encoding="utf-8") as wpm_out:
        # Iterate through all .mp3 files in the folder
        for filename in os.listdir(audio_folder):
            if filename.endswith(".mp3"):
                audio_path = os.path.join(audio_folder, filename)
                print(f"Transcribing: {filename}")

                # Transcribe the audio file using Whisper
                result = model.transcribe(audio_path)
                transcription = result["text"]
                
                # Ground truth file path
                ground_truth_file = os.path.join(audio_folder, f"{filename}.txt")
                
                # Calculate WPE if ground truth exists
                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
                        ground_truth = gt_file.read().strip()
                        wpe = calculate_wpe(transcription, ground_truth)
                else:
                    print(f"Ground truth file for {filename} not found. Skipping WPE calculation.")
                    wpe = 0  # If ground truth file doesn't exist, skip WPE calculation

                # Calculate WPM and duration
                wpm, duration, word_count = calculate_wpm(transcription, audio_path)

                # Write the results to the report files
                wpe_out.write(f"{filename}: {wpe:.4f}\n")
                wpm_out.write(f"{filename}: WPM={wpm:.2f}, Duration={duration:.2f}s, Word Count={word_count}\n")

                # Save the transcription in output folder
                output_file = os.path.join(output_folder, f"{filename}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription:\n{transcription}\n\nWPM: {wpm:.2f}\nWPE: {wpe:.4f}")
                
    print("Transcription and report generation completed.")

# Main function to initiate the process
def main():
    process_audio_files()

# Check if the script is being executed directly
if __name__ == "__main__":
    main()
