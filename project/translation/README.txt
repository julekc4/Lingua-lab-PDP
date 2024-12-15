DISCLAIMER
This portion of the project has been made using Sean Robertson's tutorial for making a sequence to sequence translation model in PyTorch.
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
While some tweaks have been made to fit it to our needs the vast majority of the code is copied verbatim from that tutorial.

USAGE:
MODEL TRAINING:
The first step to training a model using this code is to launch the main portion of this program located in main.py, it takes tab delimited data with the input language in the first column 
and the desired output language in the second column (if your data is in reverse order there is a reverse variable that needs to be set to True in the calls to functions readLangs and 
prepareData). The filepath to your data can be set in the data_fp variable at the top of the program. In the same location you should set the filepaths you'd like your finished encoder and 
decoder, as well as your checkpoints to be saved during and after training. In the section above you can set the number of epochs you'd like to train your model with and the maximum length 
of a sentence that can be translated, increasing these will have an effect on the training time. Upon running the code the model will start training and display its progress in the command 
line upon each completed epoch (or every set number of completed epoch as adjusted in the training function in print_every) it will also upon completion of the training plot the losses 
every number of epochs set in plot_every, however interrupting the training will also remove the data for that period from the plot. To resume interrupted training comment the train function 
and uncomment the resume_training function. In the current variation only one checkpoint is stored to recover a model whose training was interrupted, the quality of the model at each 
checkpoint is not evaluated so there is no guarantee that the best state of the model will be stored.

LANGUAGE PREPARATION:
If you have trained your model or have a pre-trained one ready you should run the lang_pickler script to save the Lang class objects created from your data. This should be done with the 
data the model was trained on. These functions could be integrated into the main script but it's more convenient to execute them seperately when you have a pre-trained model. 
At the top of the script you can set the filepath for your data in data_fp and the filepath you'd like the pickle language to be saved at in lang_fp.py

TRANSLATION:
Once all of the necessary files are ready (the pickled language and the trained encoder and decoder (example files are provided)) you can call the translate function in translate.py to 
translate a sentence. It takes the form:

translate(encoder, decoder, input_lang, output_lang, sentence)

Where the first four parameters should be copied verbatim since they're already defined in the script. You only need to input the string sentence containing the sentence you'd like to 
translate. Any sentence over the maximum length set in the model will throw an error.