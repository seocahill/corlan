import glob
import os
import subprocess

def run_tts_test():
    output_path = "tts_train_dir"
    ckpts = sorted([f for f in glob.glob(output_path + "/*/*.pth")])
    configs = sorted([f for f in glob.glob(output_path + "/*/*.json")])

    # Assuming the latest files are the ones to use
    test_ckpt = ckpts[-1] if ckpts else None
    test_config = configs[-1] if configs else None

    # Ensure that both files are found
    if test_ckpt and test_config:
        text_for_tts = "Cad a thabharfá"  # Replace with your desired text
        out_path = "out.wav"

        # Constructing the command
        command = f"tts --text \"{text_for_tts}\" --model_path {test_ckpt} --config_path {test_config} --out_path {out_path}"

        try:
            # Running the command
            subprocess.run(command, check=True, shell=True)
            print("TTS test run completed.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running TTS test: {e}")
    else:
        print("Checkpoint or config file not found.")

def play_audio():
    import IPython
    IPython.display.Audio("out.wav")

if __name__ == "__main__":
    run_tts_test()
    play_audio()
