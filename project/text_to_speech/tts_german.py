import subprocess
import os



def tts(sentence,out_dir='output_de'):
    filename = f"sentence{len(os.listdir(out_dir))}_de.wav"
    try:
        # Define the command
        command = [
            "tts",
            "--out_path", f"{out_dir}/{filename}",
            "--model_name", "tts_models/de/thorsten/tacotron2-DCA",
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


sentence = "Ihr naht euch wieder, schwankende Gestalten, Die früh sich einst dem trüben Blick gezeigt. Versuch ich wohl, euch diesmal festzuhalten? Fühl ich mein Herz noch jenem Wahn geneigt? Ihr drängt euch zu! nun gut, so mögt ihr walten, Wie ihr aus Dunst und Nebel um mich steigt;"
tts(sentence)
