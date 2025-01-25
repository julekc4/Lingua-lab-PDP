import subprocess
import os




def tts(sentence,out_dir='output_eng'):
    filename = f"sentence{len(os.listdir(out_dir))}.wav"
    try:
        # Define the command
        command = [
            "tts",
            "--out_path", f"{out_dir}/{filename}",
            "--model_name", "tts_models/en/ljspeech/tacotron2-DDC",
            "--text", sentence
        ]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the command succeeded
        if result.returncode == 0:
            print("Speech synthesis completed successfully!")
        else:
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("The 'tts' command was not found. Make sure Coqui TTS is installed and added to your PATH.")


