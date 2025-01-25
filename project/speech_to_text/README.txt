DISCLAIMER:

This project is a simple pipeline for converting audio files into text using OpenAI's Whisper model. 

Before the coversion, the script preprocesses audio file(s) from mp3 to wav format if needed, and performs some other 

The script supports preprocessing audio files, converting MP3 to WAV format, it is also normalised, and stripped to eliminate noises.

Afterwards, the quality of the conversion is checked if needed, and later the final result is saved as a txt file.

PREREQUISITES:

Make sure that the following dependencies are installed before running the script:

FFmpeg (required by pydub for audio processing):
sudo apt update && sudo apt install -y ffmpeg

Python Libraries:
Install the necessary Python libraries with the following command:
pip install "SpeechRecognition pydub git+https://github.com/openai/whisper.git jiwer"

HOW TO USE:

Set Up Input and Output Folders:
Place your audio files (MP3 or WAV) in the audio_files folder. Make sure that only the audio files you want to transform are in the folder
Ensure that the txt_files folder exists or let the script create it automatically.
Transcription results will be saved in txt_files/transcription_results.txt.

