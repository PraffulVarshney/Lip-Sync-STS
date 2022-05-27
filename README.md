# Lip-Sync-STS
When we translate a video from one language to another, lips get out of sync due to characteristics of different languages. So, in this project, we built a model to Sync the Lips in accordance with the Audio. To keep the project uncomplicated project was divided into two tasks-

Task 1- Translate the audio from English to Hindi.<br/>
Task 2- Sync the Lips of video according to generated Hindi audio.

## Task 1
In this task, for speech-to-speech translation, we use an automatic speech recognizer (ASR) to transcribe text from the original speech in Hindi language. We adapt neural machine translation and text-to-speech models to work for Indian languages and generate translated speech. The Task 1 proceeds in following way:<br/><br/> Video  -> English Audio  -> English Text  -> Hindi Text  -> Hindi Audio.<br/><br/>
The input given is the video with audio in English language. We extracted English audio from video using Moviepy library. Then converted this audio to text using Deepvoice 2 model of Pytorch. Then translated this English text to Hindi Text using a language translation model of huggingface. There was limit of 5000 bytes for this model, so we broke the text into several chunks of 2000 bytes and then applied the model. After that using Google API, we generated voice from the translated Hindi text and will merge it using Lip GAN. We adopt this approach to achieve high quality text-to-speech synthesis in our target language.<br/>
https://colab.research.google.com/drive/1scp7LqkbFx5QghIPU3W4F-7771BzNV9S?usp=sharing

<a href="url"><img src="https://user-images.githubusercontent.com/79749572/167295444-d010a382-8026-460e-bcc2-83eb76225dad.png" width="720" height = "350">

## Task 2
In this task we are given with a source or input video and a translated audio speech in Hindi. The translated audio is created using the English audio or speech from given input video file. Our task is to generate a lip-synced video having speech language as Hindi using these two inputs. <br/>
This task consists of a GAN network having a generator to generate lip-synced frames and discriminator to check whether lip-synced occurred or not. Both are discussed in detail later on 

## Model Formulation
In a nutshell, our setup contains two networks, a generator G that generates faces by conditioning on audio inputs and a discriminator D that tests whether the generated face and the input audio are in sync. By training these networks together, the generator G learns to create photo-realistic faces that are accurately in sync with the given input audio.
  
<a href="url"><img src="https://user-images.githubusercontent.com/79749572/167295390-8d53d96d-462f-4fd2-995e-1f198fa3af08.png" width="720">

### Generator:-
The generator network contains three branches: <br/><br/>
(i) The Face Encoder<br/>
During the training process of the generator , a face image of random pose and its corresponding audio segment is given as input and the generator is expected to morph the lip shape. Along with the random identity face image I, we also provide the desired pose information of the ground-truth as input to the face encoder. We mask the lower half of the ground truth face image and concatenate it channel-wise with I.
The masked ground truth image provides the network with information about the target pose while ensuring that the network never gets any information about the ground truth lip shape. Thus, our final input to the face encoder , a regular CNN, is a HxHx6 image.<br/><br/>
(ii) Audio Encoder<br/>
The audio encoder is a standard CNN that takes a Mel-frequency cepstral coefficient (MFCC) heatmap of size MxTx1 and creates an audio embedding of size h. The audio embedding is concatenated with the face embedding to produce a joint audio-visual embedding of size 2xh.<br/><br/>
(iii) Face Decoder<br/>
This branch produces a lip-synchronized face from the joint audio-visual embedding by inpainting the masked region of the input image with an appropriate mouth shape. It contains a series of residual blocks with a few intermediate deconvolutional layers. The output layer of the Face decoder is a sigmoid activated 1x1 convolutional layer with 3 filters, resulting in a face image of HxHx3.<br/>

### Discriminator:-
We used L2 reconstruction loss for the generator that generated satisfactory talking faces, employing strong additional supervision can help the generator learn robust, accurate phoneme viseme mappings and make the facial movements more natural. We are directly testing whether the generated face synchronizes with the audio provides a stronger supervisory signal to the generator network. Accordingly, we create a network that encodes an input face and audio into fixed representations and computes the L2 distance d between them. The face encoder and audio encoder are the same as used in the generator network. The discriminator learns to detect synchronization by minimizing the following contrastive loss. 
 
<a href="url"><img src="https://user-images.githubusercontent.com/79749572/167295701-e0b9680d-da1f-477e-becf-5b29ed6524c8.png" width="500" height = "200">

## Result
### Input Video
  
  <a href="url"><video src="https://user-images.githubusercontent.com/96406063/167405376-57056631-cae9-49d9-9aaf-fb1419fc5b5f.mp4"><br/>
  

  
### Translated Audio
  
  [hindi_audio.zip](https://github.com/ariesiitr/Lip-Sync-STS/files/8651500/hindi_audio.zip)<br/>

  
### Final Output

  <a href="url"><video src="https://user-images.githubusercontent.com/96406063/167412441-83e9c9f0-c3f2-4b27-aed7-e5a482037fc5.mp4"><br/>

 
## Applications

<a href="url"><img src="https://user-images.githubusercontent.com/79749572/167292981-27bfc583-c9e5-44ef-9483-23250d6a861b.png" width="800">  

### Reference Papers
https://arxiv.org/pdf/2003.00418v1.pdf<br/>
https://arxiv.org/format/1611.01599<br/>
https://arxiv.org/pdf/2008.10010v1.pdf
  
### Contributers
  
  <a href="https://github.com/praffulv-225">Prafful Varshney</a><br/>
  <a href="https://github.com/raosatyam">Satyam Yadav</a><br/>
  <a href="https://github.com/nishir53">Nishir Agrawal</a>

















