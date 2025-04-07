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
cleaned_audio_folder = "cleaned_audio"  # New cleaned audio folder

# Ensure output and report folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
os.makedirs(cleaned_audio_folder, exist_ok=True)

# Combined report file
combined_report_file = os.path.join(report_folder, "combined_performance_report.txt")


# Function to preprocess audio
def audio_preprocess(input_path, output_path_mp3, target_sr=16000):
    """
    General-purpose audio cleaner:
    - Converts to mono
    - Normalizes
    - Applies noise reduction
    - Resamples
    - Exports to .mp3
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    y = y / np.max(np.abs(y))  # Normalize
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    if sr != target_sr:
        y_resampled = librosa.resample(y_denoised, orig_sr=sr, target_sr=target_sr)
    else:
        y_resampled = y_denoised

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        sf.write(tmp_wav.name, y_resampled, target_sr)
        tmp_wav_path = tmp_wav.name

    audio = AudioSegment.from_wav(tmp_wav_path)
    audio.export(output_path_mp3, format="mp3")
    os.remove(tmp_wav_path)


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

                cpu_before, mem_before = get_resource_usage()
                start_time = time.time()

                # ====== OPTIONAL AUDIO PREPROCESSING ======
                # To disable preprocessing, comment out the next 3 lines and uncomment the one below them
                cleaned_audio_path = os.path.join(cleaned_audio_folder, f"cleaned_{os.path.splitext(filename)[0]}.mp3")
                audio_preprocess(audio_path, cleaned_audio_path)
                audio_to_transcribe = cleaned_audio_path

                # Use original (uncomment to skip cleaning)
                # audio_to_transcribe = audio_path

                # Transcribe the audio
                result = model.transcribe(audio_to_transcribe)
                transcription = result["text"]

                cpu_after, mem_after = get_resource_usage()
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Ground truth path
                ground_truth_folder = "audio_text"
                ground_truth_file = os.path.join(ground_truth_folder, f"{filename}.txt")

                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
                        ground_truth = gt_file.read().strip()
                        wpe = calculate_wpe(transcription, ground_truth)
                else:
                    print(f"⚠️ Ground truth file for {filename} not found. Skipping WPE calculation.")
                    wpe = "N/A"

                wpm, duration, word_count = calculate_wpm(transcription, audio_to_transcribe)

                # Save transcription
                output_file = os.path.join(output_folder, f"{filename}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription:\n{transcription}\n\nWPM: {wpm:.2f}\nWPE: {wpe}")

                # Report
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
