Lip-Sync Speech-to-Speech
(ArIES 2nd year recruitment project)
When we translate a video from one language to another, lips get out of sync due to characteristics of different languages. So, in this project, we had to Sync the Lips in accordance with the Audio. The project was divided into two tasks-
Task 1- Translate the audio from English to Hindi.
Task 2- Sync the Lips of video according to generated Hindi audio.


Task 1 – 
In this task, for speech-to-speech translation, we use an automatic speech recognizer (ASR) to transcribe text from the original speech in Hindi language. We adapt neural machine translation and text-to-speech models to work for Indian languages and generate translated speech. 
The Task 1 proceeds in following way:
Video    English Audio    English Text    Hindi Text    Hindi Audio
The input given is the video with audio in English language. We extracted English audio from video using Moviepy library. Then converted this audio to text using Deepvoice 2 model of Pytorch. Then translated this English text to Hindi Text using a language translation model of huggingface. There was limit of 5000 bytes for this model, so we broke the text into chunks of 2000 bytes and then applied the model. After that using Google API, generated voice from the translated Hindi text and merged it to the video. We adopt this approach to achieve high quality text-to-speech synthesis in our target language.

Task 2 –
In this task we are given with a source or input video and a translated audio speech in Hindi. The translated audio is created using the English audio or speech from given input video file. Our task is to generate a lip-synced video having speech language as Hindi using these two inputs. 
This task consists of a GAN network having a generator to generate lip-synced frames and discriminator to check whether lip-synced occurred or not. 

Model Formulation
Given a face image I containing a subject identity and a speech A divided into a sequence of speech segments {A1,A2, ...Ak } , we would like to design a model G, that generates a sequence of frames {S1, S2, ...Sk } that contains the face speaking the audio A with proper lip synchronization. 
In a nutshell, our setup contains two networks, a generator G that generates faces by conditioning on audio inputs and a discriminator D that tests whether the generated face and the input audio are in sync. By training these networks together, the generator G learns to create photo-realistic faces that are accurately in sync with the given input audio.

Generator
The generator network contains three branches: 
(i) Face encoder (ii) Audio encoder (iii) Face Decoder.
The Face Encoder
During the training process of the generator, a face image of random pose and its corresponding audio segment is given as input and the generator is expected to morph the lip shape. Along with the random identity face image I, we also provide the desired pose information of the ground-truth as input to the face encoder. We mask the lower half of the ground truth face image and concatenate it channel-wise with I. 
The masked ground truth image provides the network with information about the target pose while ensuring that the network never gets any information about the ground truth lip shape. Thus, our final input to the face encoder, a regular CNN, is a HxHx6 image. The encoder consists of a series of residual blocks with intermediate down-sampling layers, and it embeds the given input image into a face embedding of size h
Audio Encoder
The audio encoder is a standard CNN that takes a Mel-frequency cepstral coefficient (MFCC) heatmap of size MxTx1 and creates an audio embedding of size h. The audio embedding is concatenated with the face embedding to produce a joint audio-visual embedding of size 2xh.
Face Decoder
This branch produces a lip-synchronized face from the joint audio-visual embedding by inpainting the masked region of the input image with an appropriate mouth shape. It contains a series of residual blocks with a few intermediate deconvolutional layers. The output layer of the Face decoder is a sigmoid activated 1x1 convolutional layer with 3 filters, resulting in a face image of HxHx3. We employ 6 skip connections, one after every deconvolution operation to ensure that input facial features are preserved by the decoder while generating the face.


Discriminator
While using only an L2 reconstruction loss for the generator can generate satisfactory talking faces, employing strong additional supervision can help the generator learn robust, accurate phoneme viseme mappings and make the facial movements more natural. We are directly testing whether the generated face synchronizes with the audio provides a stronger supervisory signal to the generator network. Accordingly, we create a network that encodes an input face and audio into fixed representations and computes the L2 distance d between them. The face encoder and audio encoder are the same as used in the generator network.
We randomly select a T millisecond window from an input video sample and extract its corresponding audio segment A, resampled at a frequency F Hz. We choose the middle video frame S in this window as the desired ground-truth and mask the mouth region (assumed to be the lower-half of the image) of a person in the ground truth frame to get Sm. We also sample a negative frame S ′ , i.e., a frame outside this window which is expected to not be in sync with the chosen audio segment A.
At each training batch to the generator, the unsynced face S ′ concatenated channel wise with the masked ground truth face Sm and the target audio segment A is provided as the input. The generator is expected to generate the synced face G([S ′ ; Sm],A) ≈ S. Each training batch to the discriminator contains three types of samples: (i) Synthetic samples from the generator (G(S ′ ,A),A);yi = 1, (ii) Actual frames synced with audio (S,A);yi = 0 and (iii) Actual frames out of sync with audio (S ′ ,A);yi = 1. The third sample type is particularly important to force the discriminator to take into account the lip synchronization factor while classifying a given input pair as real / synthetic. Without the third type of sample, the discriminator would simply be able to ignore the audio input and make its decision solely on the quality of the image. The discriminator learns to detect synchronization by minimizing the following contrastive loss:
 

Applications
Educational videos
A large amount of online educational content is present in English in the form of video lectures. Dubbing these videos with just speech-to-speech systems creates a visual discrepancy between the lip motion and the dubbed audio. However, with the help of our face-to-face translation system, it is possible to automatically translate such educational video content and ensure lip synchronization.
Television news and interviews
Automatic Face-to-Face Translation systems can potentially allow viewers to access and consume important information from across the globe irrespective of the underlying language. For example, a Hindi or German viewer can watch an English interview of Obama in the language of his/her choice with lip synchronization.
Movie dubbing
Movies are generally dubbed by dubbing artists manually. The dubbed audio is then overlaid to the original video. This causes the actors’ lips to be out of sync with the audio, thus affecting the viewer experience. Our pipeline can be used to automate this process of lip-syncing. That is, given a movie scene in a particular language our system can potentially be used to dub it to a different language.


Contributers-
Prafful Varshney
Nishir Agrawal
Satyam Yadav
Ayushi Shikha 
Sristi Mishra

Mentored By-
Navtej Mishra

Reference - http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation

 
