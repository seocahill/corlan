import subprocess
import re

def split_mp3(input_file, silence_thresh=-30, silence_duration=1):
    # Generate the silence detection log file
    log_file = 'silence.log'
    subprocess.call(f'ffmpeg -i "{input_file}" -af silencedetect=noise={silence_thresh}dB:d={silence_duration} -f null - 2> {log_file}', shell=True)

    # Read and parse the log file
    with open(log_file, 'r') as file:
        log_content = file.read()

    # Regular expression to find timestamps
    pattern = r'silence_start: (\d+\.\d+)|silence_end: (\d+\.\d+)'
    timestamps = re.findall(pattern, log_content)

    # Process timestamps to get start and end times of non-silent segments
    segments = []
    last_silence_end = 0.0

    for start, end in timestamps:
        if start:
            # If there is a start, use the last end as the start of the segment
            segments.append((last_silence_end, float(start)))
        elif end:
            # Update the last end time
            last_silence_end = float(end)

    # Split the MP3 file using the timestamps
    for i, (start, end) in enumerate(segments):
        output_file = f"output_segment_{i+1}.mp3"
        subprocess.call(f'ffmpeg -i "{input_file}" -ss {start} -to {end} -c copy "{output_file}"', shell=True)
        print(f"Segment {i+1} saved as {output_file}")


# Example usage
split_mp3("/Users/seocahill/Desktop/C1.01-06.mp3")
