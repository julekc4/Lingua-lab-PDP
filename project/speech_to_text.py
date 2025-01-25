#sudo apt update && sudo apt install -y ffmpeg
#pip install "SpeechRecognition pydub git+https://github.com/openai/whisper.git jiwer"

import os
from pydub import AudioSegment, effects
from jiwer import wer, cer
import whisper
import soundfile as sf

whisper_model = whisper.load_model("base")

def preprocess_audio(file_path):
    #print(f"Preprocessing audio: {file_path}")
    audio = AudioSegment.from_file(file_path)

    audio = effects.normalize(audio)

    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)

    audio = audio.strip_silence(silence_thresh=-40, padding=200)

    audio.export(file_path, format="wav")

    return file_path


def convert_to_wav(file_path):
    if file_path.endswith(".mp3"):
        wav_path = file_path.replace(".mp3", ".wav")
        sound = AudioSegment.from_mp3(file_path)
        sound.export(wav_path, format="wav")
        return wav_path
    return file_path

def whisper_method(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def calculate_accuracy(ground_truth, predicted): #not needed 
    word_error_rate = wer(ground_truth, predicted)
    char_error_rate = cer(ground_truth, predicted)
    return word_error_rate, char_error_rate

def compare_methods(audio_path, ground_truth, output_file): #accuracy check not used only in report 
    #uncomment the green lines in case of wanting to see the accuracy as well
    with open(output_file, "a") as f:
        #f.write(f"\nProcessing file: {audio_path}\n")
        #f.write("\nRunning Whisper...\n")
        whisper_transcription = whisper_method(audio_path)
        f.write(whisper_transcription)
        #f.write("\nComparison Results:\n")
        #f.write("-" * 50 + "\n")
        #f.write("Ground Truth:\n")
        #f.write(ground_truth + "\n\n")
        #f.write("Whisper Output:\n")
        #f.write(whisper_transcription + "\n")
        #f.write("-" * 50 + "\n")

        #f.write("\nAccuracy Evaluation:\n")
        #whisper_wer, whisper_cer = calculate_accuracy(ground_truth, whisper_transcription)
        #f.write(f"Whisper - WER: {whisper_wer:.2f}, CER: {whisper_cer:.2f}\n")

def main():
    input_folder = "./project/speech_to_text/audio_files"  
    output_folder = "./project/output"    
    output_file = os.path.join(output_folder, "transcription_results.txt")

    os.makedirs(output_folder, exist_ok=True)


    if os.path.exists(output_file):
        os.remove(output_file)
    #only needed to check the accuracy of the speech to text model
    ground_truth = (
        "The stale smell of old beer lingers. It takes heat to bring out the odor. "
        "A cold dip restores health and zest. A salt pickle tastes fine with ham. "
        "Tacos Al Pastor are my favorite. A zestful food is the hot-cross bun."
    )

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3") or file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            
            wav_path = convert_to_wav(file_path)
            
            file_path = preprocess_audio(wav_path)
            
            compare_methods(file_path, ground_truth, output_file)

if __name__ == "__main__":
    main()
