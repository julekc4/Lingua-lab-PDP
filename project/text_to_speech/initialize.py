
#FIRST DO: pip install TTS and resolve dependencies!!

import subprocess

def initialize():
    try:
        # Run the git clone command
        result = subprocess.run(
            ["pip", "install", "TTS"],
            capture_output=True,
            text=True,
        )

        # Check if the command was successful
        if result.returncode == 0:
            print("Installed successfully!")
        else:
            print(f"Error installing library: {result.stderr}")
    except FileNotFoundError:
        print("Git is not installed or not found in the system PATH.")


    try:
        # Run the git clone command
        result = subprocess.run(
            ["pip", "install", "python-espeak-ng"],
            capture_output=True,
            text=True,
        )

        # Check if the command was successful
        if result.returncode == 0:
            print("Installed successfully!")
        else:
            print(f"Error installing library: {result.stderr}")
    except FileNotFoundError:
        print("Git is not installed or not found in the system PATH.")



def list_models():
    try:
        # Run the tts command
        result = subprocess.run(["tts", "--list_models"], capture_output=True, text=True)

        # Print the list of models
        if result.returncode == 0:
            print("Available Models:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except FileNotFoundError:
        print("The 'tts' command was not found. Make sure Coqui TTS is installed and in your PATH.")


list_models()
