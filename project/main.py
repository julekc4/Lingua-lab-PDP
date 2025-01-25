import translate as translate
import speech_to_text as stt

trans_inp = "./output/transcription_results.txt"
trans_out = "./output/translation_results.txt"
out_lang = "pl"

encoder_fp = "./translation/en-de_trained_model/EN-DE-Encoder.pt"
decoder_fp = "./translation/en-de_trained_model/EN-DE-Decoder.pt"
lang_fp = "./translation/en-de_trained_model/eng-ger.pkl"

stt.main()
translate(trans_inp, trans_out, out_lang)