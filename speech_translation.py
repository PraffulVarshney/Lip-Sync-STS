
import moviepy.editor as mp
import speech_recognition as sr
import numpy as np
import pandas as pd
import scipy 
import librosa
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS


# take the input english video
my_clip = mp.VideoFileClip(r"input_video")

#Extracting English audio from Video
my_clip.audio.write_audiofile(r"english_audio.wav")
filename = "english_audio.wav"
r = sr.Recognizer()

# open the audio file
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text1 = r.recognize_google(audio_data)
   # print(text1)
file = open("english_sub.txt", "w") 
file.write(text1) 
file.close()

#loading the tokenizer and the model
#loading pretrained model for translation from English to Hindi

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def translator(text):
  # function to translate english text to hindi
  input_ids = tokenizer.encode(text, return_tensors="pt", padding=True)
  outputs = model.generate(input_ids)
  decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  
  return decoded_text


#text you want to translate
#Using the model for translation
texts = open("english_sub.txt", "r") 
print(texts)
#file.write(text1) 
#file.close()

for text in texts:
 # print("English Text: ", text)
  translator(text) 

file = open("hindi_sub.txt", "w") 
file.write(translator(text)) 
file.close()

# Translating generated text using Google API
hindi = translator(text)
obj = gTTS(text = hindi, slow = False, lang = 'hi')
obj.save('hindi_audio.wav')

# Final output Hindi audio as Hindi_audio.wav