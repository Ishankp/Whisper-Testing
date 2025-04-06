import whisper
import os
import librosa
import Levenshtein
import psutil
import time

# Folders
audio_folder = "audio_folder"
output_folder = "output_folder"
report_folder = "report"

# Ensure output and report folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

# Combined report file
combined_report_file = os.path.join(report_folder, "combined_performance_report.txt")

#Function to perform external audio preprocessing
##def audio_preprocess ():

# Function to calculate WPE (Word Per Error Rate)
def calculate_wpe(transcription, ground_truth):
    distance = Levenshtein.distance(transcription.lower(), ground_truth.lower())  # Case-insensitive
    wpe = distance / len(ground_truth.split()) if len(ground_truth.split()) > 0 else None
    return wpe

# Function to calculate WPM (Words Per Minute)
def calculate_wpm(transcription, audio_path):
    duration = librosa.get_duration(path=audio_path)  # Get duration in seconds
    duration_minutes = duration / 60  # Convert to minutes
    word_count = len(transcription.split())
    wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    return wpm, duration, word_count

# Function to measure CPU & Memory Usage
def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU usage over 1 second
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return cpu_usage, memory_usage

# Function to process each audio file
def process_audio_files():
    model = whisper.load_model("base")

    with open(combined_report_file, "w", encoding="utf-8") as report_out:
        report_out.write("🔹 Combined Performance Report 🔹\n")
        report_out.write("=" * 50 + "\n")

        for filename in os.listdir(audio_folder):
            if filename.endswith(".mp3"):
                audio_path = os.path.join(audio_folder, filename)
                print(f"🎙️ Transcribing: {filename}")

                # Record CPU & Memory usage before processing
                cpu_before, mem_before = get_resource_usage()
                start_time = time.time()

                # Perform external audio preprocessing

                # Transcribe the audio file using Whisper
                result = model.transcribe(audio_path)
                transcription = result["text"]

                # Record CPU & Memory usage after processing
                cpu_after, mem_after = get_resource_usage()
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Ground truth file path
                ground_truth_folder = "audio_text"  # Define the correct folder for ground truth files
                ground_truth_file = os.path.join(ground_truth_folder, f"{filename}.txt")  # Look for the file in the correct folder


                # Calculate WPE if ground truth exists
                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
                        ground_truth = gt_file.read().strip()
                        wpe = calculate_wpe(transcription, ground_truth)
                else:
                    print(f"⚠️ Ground truth file for {filename} not found. Skipping WPE calculation.")
                    wpe = "N/A"

                # Calculate WPM and duration
                wpm, duration, word_count = calculate_wpm(transcription, audio_path)

                # Save the transcription in output folder
                output_file = os.path.join(output_folder, f"{filename}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription:\n{transcription}\n\nWPM: {wpm:.2f}\nWPE: {wpe}")

                # Write all metrics to the combined report file
                report_out.write(f"🔹 {filename}\n")
                report_out.write(f"   🔸 WPE (Word Per Error Rate): {wpe}\n")
                report_out.write(f"   🔸 WPM (Words Per Minute): {wpm:.2f}\n")
                report_out.write(f"   🔸 Audio Duration: {duration:.2f} seconds\n")
                report_out.write(f"   🔸 Word Count: {word_count}\n")
                report_out.write(f"   🔸 CPU Usage: Before: {cpu_before:.2f}% | After: {cpu_after:.2f}%\n")
                report_out.write(f"   🔸 Memory Usage: Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB\n")
                report_out.write(f"   🔸 Processing Time: {elapsed_time:.2f} seconds\n")
                report_out.write("=" * 50 + "\n")

                print(f"✅ Finished: {filename} | Time: {elapsed_time:.2f}s | CPU: {cpu_after:.2f}% | Mem: {mem_after:.2f}MB")

    print("🚀 Transcription, performance tracking, and report generation completed.")

# Main function to initiate the process
def main():
    process_audio_files()

if __name__ == "__main__":
    main()
