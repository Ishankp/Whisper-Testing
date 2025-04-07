import os

# Folders and paths
report_folder = "report"
combined_report_file = os.path.join(report_folder, "combined_performance_report.txt")

# Function to extract relevant stats from each report
def extract_stats_from_report(report_file):
    stats = {
        "WPE": None,
        "WPM": None,
        "audio_duration": None,
        "word_count": None,
        "preprocessing_time": None,
        "transcription_time": None,
        "total_processing_time": None,
        "cpu_before": None,
        "cpu_after": None,
        "memory_before": None,
        "memory_after": None
    }

    with open(report_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if "WPE" in line:
                stats["WPE"] = float(line.split(":")[1].strip())
            elif "WPM" in line:
                stats["WPM"] = float(line.split(":")[1].strip())
            elif "Audio Duration" in line:
                stats["audio_duration"] = float(line.split(":")[1].strip().split()[0])
            elif "Word Count" in line:
                stats["word_count"] = int(line.split(":")[1].strip())
            elif "Preprocessing Time" in line:
                stats["preprocessing_time"] = float(line.split(":")[1].strip().split()[0])
            elif "Transcription Time" in line:
                stats["transcription_time"] = float(line.split(":")[1].strip().split()[0])
            elif "Total Processing Time" in line:
                stats["total_processing_time"] = float(line.split(":")[1].strip().split()[0])
            elif "CPU Usage" in line:
                cpu_stats = line.split("|")
                stats["cpu_before"] = float(cpu_stats[0].split(":")[1].strip().split()[0])
                stats["cpu_after"] = float(cpu_stats[1].split(":")[1].strip().split()[0])
            elif "Memory Usage" in line:
                memory_stats = line.split("|")
                stats["memory_before"] = float(memory_stats[0].split(":")[1].strip().split()[0])
                stats["memory_after"] = float(memory_stats[1].split(":")[1].strip().split()[0])

    return stats


# Function to calculate the average of stats
def calculate_average(stats_list):
    averages = {
        "WPE": 0,
        "WPM": 0,
        "audio_duration": 0,
        "word_count": 0,
        "preprocessing_time": 0,
        "transcription_time": 0,
        "total_processing_time": 0,
        "cpu_before": 0,
        "cpu_after": 0,
        "memory_before": 0,
        "memory_after": 0
    }

    num_reports = len(stats_list)

    for stats in stats_list:
        for key in averages:
            if stats[key] is not None:
                averages[key] += stats[key]

    # Calculate averages
    for key in averages:
        averages[key] /= num_reports

    return averages


# Function to compile the report and print the averages
def compile_report():
    batch_stats = []
    real_time_stats = []

    # Go through each report file and extract stats
    for filename in os.listdir(report_folder):
        if filename.endswith(".txt") and filename != "combined_performance_report.txt":
            report_file = os.path.join(report_folder, filename)
            stats = extract_stats_from_report(report_file)

            # Categorize by batch or real-time (based on the filename)
            if "Real-Time" in filename:
                real_time_stats.append(stats)
            elif "Batch" in filename:
                batch_stats.append(stats)

    # Calculate averages for batch and real-time
    batch_avg = calculate_average(batch_stats)
    real_time_avg = calculate_average(real_time_stats)

    # Output the averages
    with open(combined_report_file, "a", encoding="utf-8") as f:
        f.write("\n\n🔹 Average Performance Report 🔹\n")
        f.write("=" * 50 + "\n")

        # Batch averages
        f.write("\n🔹 Batch Transcription Averages\n")
        for key, value in batch_avg.items():
            f.write(f"   🔸 {key}: {value:.2f}\n")

        # Real-time averages
        f.write("\n🔹 Real-Time Transcription Averages\n")
        for key, value in real_time_avg.items():
            f.write(f"   🔸 {key}: {value:.2f}\n")

        f.write("=" * 50 + "\n")

    print("✅ Averages compiled and written to the combined report.")


if __name__ == "__main__":
    compile_report()
