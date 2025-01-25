"""
Created using elements from a tutorial to RNN translation in PyTorch by Sean Robertson
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from nltk import tokenize, download

download('punkt_tab')

#MBART implementation
def translate(fp_in, fp_out, lang):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    lang_dict = {"pl": "pl_PL", "de": "de_DE", "ru": "ru_RU"}
    
    if lang not in lang_dict:
        return "Incorrect language selected"

    with open(fp_in, 'r') as f:
        sentences = f.read()

    output = []
    sentences = tokenize.sent_tokenize(sentences)
    for sentence in sentences:
        tokenizer.src_lang = "en_XX"
        encoded_hi = tokenizer(sentence, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_hi,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang_dict[lang]]
        )
        output.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    with open(fp_out, "w", encoding="utf-8") as f:
        i = 0
        for sentence in output:
            f.write(sentence + " ")
            i += 1
            if i % 3 == 0:
                f.write("\n")
    

#translate(encoder, decoder, input_lang, output_lang, "test_file.txt", "test_file_out.txt")

