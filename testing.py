import whisper
import os
import librosa
import Levenshtein
import psutil
import time
import noisereduce as nr
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


# Folders
audio_folder = "audio_folder"
output_folder = "output_folder"
report_folder = "report"
cleaned_audio_folder = "cleaned_audio"

# Ensure output, report, and cleaned audio folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
os.makedirs(cleaned_audio_folder, exist_ok=True)

# Combined report file
combined_report_file = os.path.join(report_folder, "combined_performance_report.txt")

# Function to perform external audio preprocessing (e.g., noise reduction, normalization)
def audio_preprocess(audio_path, cleaned_audio_path):
    print(f"🎧 Preprocessing: {audio_path}")
    # Load the audio file using pydub
    audio = AudioSegment.from_mp3(audio_path)
    # Normalize the audio (brings the volume to a consistent level)
    audio = audio.normalize()
    # Export the processed audio to a new file
    audio.export(cleaned_audio_path, format="mp3")
    print(f"✅ Preprocessed audio saved: {cleaned_audio_path}")

# Function to calculate WPE (Word Per Error Rate)
def calculate_wpe(transcription, ground_truth):
    distance = Levenshtein.distance(transcription.lower(), ground_truth.lower())
    wpe = distance / len(ground_truth.split()) if len(ground_truth.split()) > 0 else None
    return wpe

# Function to calculate WPM (Words Per Minute)
def calculate_wpm(transcription, audio_path):
    duration = librosa.get_duration(path=audio_path)
    duration_minutes = duration / 60
    word_count = len(transcription.split())
    wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    return wpm, duration, word_count

# Function to measure CPU & Memory Usage
def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = process.memory_info().rss / (1024 * 1024)
    return cpu_usage, memory_usage

# Function to process each audio file in real-time (simulating real-time transcription)
def process_audio_real_time():
    model = whisper.load_model("base")
    with open(combined_report_file, "w", encoding="utf-8") as report_out:
        report_out.write("🔹 Combined Performance Report 🔹\n")
        report_out.write("=" * 50 + "\n")
        for filename in os.listdir(audio_folder):
            if filename.endswith(".mp3"):
                audio_path = os.path.join(audio_folder, filename)
                cleaned_audio_path = os.path.join(cleaned_audio_folder, filename)
                real_time_output_file = os.path.join(output_folder, f"R{filename}.txt")
                print(f"🎙️ Real-Time Transcribing: {filename}")
                
                # Time the audio preprocessing step
                pre_start = time.time()
                # Preprocess audio (optional – comment out to disable preprocessing)
                audio_preprocess(audio_path, cleaned_audio_path)
                pre_end = time.time()
                preprocess_time = pre_end - pre_start

                # Record CPU & Memory usage before transcription processing
                cpu_before, mem_before = get_resource_usage()
                trans_start = time.time()
                
                # Transcribe the cleaned audio file using Whisper
                result = model.transcribe(cleaned_audio_path)
                transcription = result["text"]
                
                trans_end = time.time()
                transcription_time = trans_end - trans_start
                
                total_time = preprocess_time + transcription_time

                # Record CPU & Memory usage after transcription processing
                cpu_after, mem_after = get_resource_usage()
                
                # Ground truth file path (if available)
                ground_truth_folder = "audio_text"
                ground_truth_file = os.path.join(ground_truth_folder, f"{filename}.txt")
                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
                        ground_truth = gt_file.read().strip()
                        wpe = calculate_wpe(transcription, ground_truth)
                else:
                    print(f"⚠️ Ground truth file for {filename} not found. Skipping WPE calculation.")
                    wpe = "N/A"
                
                # Calculate WPM and duration using the cleaned audio
                wpm, duration, word_count = calculate_wpm(transcription, cleaned_audio_path)
                
                # Save the transcription in the real-time output file
                with open(real_time_output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription:\n{transcription}\n\nWPM: {wpm:.2f}\nWPE: {wpe}\n")
                    f.write(f"Preprocessing Time: {preprocess_time:.2f}s\n")
                    f.write(f"Transcription Time: {transcription_time:.2f}s\n")
                    f.write(f"Total Processing Time: {total_time:.2f}s\n")
                
                # Write performance metrics to the combined report
                report_out.write(f"🔹 {filename} (Real-Time)\n")
                report_out.write(f"   🔸 WPE (Word Per Error Rate): {wpe}\n")
                report_out.write(f"   🔸 WPM (Words Per Minute): {wpm:.2f}\n")
                report_out.write(f"   🔸 Audio Duration: {duration:.2f} seconds\n")
                report_out.write(f"   🔸 Word Count: {word_count}\n")
                report_out.write(f"   🔸 Preprocessing Time: {preprocess_time:.2f} seconds\n")
                report_out.write(f"   🔸 Transcription Time: {transcription_time:.2f} seconds\n")
                report_out.write(f"   🔸 Total Processing Time: {total_time:.2f} seconds\n")
                report_out.write(f"   🔸 CPU Usage: Before: {cpu_before:.2f}% | After: {cpu_after:.2f}%\n")
                report_out.write(f"   🔸 Memory Usage: Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB\n")
                report_out.write("=" * 50 + "\n")
                print(f"✅ Finished Real-Time: {filename} | Preprocess: {preprocess_time:.2f}s | Transcribe: {transcription_time:.2f}s | Total: {total_time:.2f}s")
    print("🚀 Real-Time Transcription completed.")

# Function to process each audio file in batch (processing the whole file at once)
def process_audio_files():
    model = whisper.load_model("base")
    with open(combined_report_file, "a", encoding="utf-8") as report_out:
        report_out.write("🔹 Batch Transcription Results\n")
        report_out.write("=" * 50 + "\n")
        for filename in os.listdir(audio_folder):
            if filename.endswith(".mp3"):
                audio_path = os.path.join(audio_folder, filename)
                batch_output_file = os.path.join(output_folder, f"{filename}.txt")
                print(f"🎙️ Batch Transcribing: {filename}")
                
                # Record CPU & Memory usage before processing
                cpu_before, mem_before = get_resource_usage()
                start_time = time.time()
                
                # Transcribe the original audio file using Whisper
                result = model.transcribe(audio_path)
                transcription = result["text"]
                
                cpu_after, mem_after = get_resource_usage()
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Ground truth file path (if available)
                ground_truth_folder = "audio_text"
                ground_truth_file = os.path.join(ground_truth_folder, f"{filename}.txt")
                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
                        ground_truth = gt_file.read().strip()
                        wpe = calculate_wpe(transcription, ground_truth)
                else:
                    print(f"⚠️ Ground truth file for {filename} not found. Skipping WPE calculation.")
                    wpe = "N/A"
                
                # Calculate WPM and duration using the original audio file
                wpm, duration, word_count = calculate_wpm(transcription, audio_path)
                
                # Save the transcription in the batch output file
                with open(batch_output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription:\n{transcription}\n\nWPM: {wpm:.2f}\nWPE: {wpe}")
                
                # Write performance metrics to the combined report
                report_out.write(f"🔹 {filename} (Batch)\n")
                report_out.write(f"   🔸 WPE (Word Per Error Rate): {wpe}\n")
                report_out.write(f"   🔸 WPM (Words Per Minute): {wpm:.2f}\n")
                report_out.write(f"   🔸 Audio Duration: {duration:.2f} seconds\n")
                report_out.write(f"   🔸 Word Count: {word_count}\n")
                report_out.write(f"   🔸 CPU Usage: Before: {cpu_before:.2f}% | After: {cpu_after:.2f}%\n")
                report_out.write(f"   🔸 Memory Usage: Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB\n")
                report_out.write(f"   🔸 Processing Time: {elapsed_time:.2f} seconds\n")
                report_out.write("=" * 50 + "\n")
                print(f"✅ Finished Batch: {filename} | Time: {elapsed_time:.2f}s | CPU: {cpu_after:.2f}% | Mem: {mem_after:.2f}MB")
    print("🚀 Batch Transcription completed.")

# Main function to run both real-time and batch transcription
def main():
    print("🚀 Starting Real-Time Transcription...")
    process_audio_real_time()
    print("🚀 Starting Batch Transcription...")
    process_audio_files()

if __name__ == "__main__":
    main()
