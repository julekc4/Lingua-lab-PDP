import translate as tran
import speech_to_text as stt
import tts_polish as ttspl

trans_in = "./project/output/transcription_results.txt"
trans_out = "./project/output/translation_results.txt"
out_lang = "pl"

stt.main()
tran.translate(trans_in, trans_out, out_lang)

with open(trans_out, "r", encoding='utf-8') as f:
    tts_in = f.read()
tts_out = "./project/text_to_speech/output_pl"

ttspl.tts(tts_in, tts_out)