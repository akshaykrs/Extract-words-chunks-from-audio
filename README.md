# Extract-words-chunks-from-audio

In todays senarios we are able to get chunks according to sentences from audio files using srt. But getting chunks if you don't have any srt with you, If you only have audio file with no srt and you want to get words out of it (i.e. each word in a separate chunk) is a tedious task and requires DSP (Digital Signal Processing) knowledge. This is what I am going to show you. 


# Pre-requisites:
1. Python 3.7
2. Librosa
3. Pydub
4. Matplotlib
5. Jupyter Notebook

In this I am not doing any preprocessing like noise reduction and all. Since, you can find many things for it. 

# STEP 1: Prepare directory:

First make a folder which will contain main audio file from which you have to get chunks. also put your code file in same directory for simplicity. 

# Step 2: Import libraries:

I am using Librosa and pydub for this task. So, If you have these libraries installed It's good. But, If not then execute these commands on jupyter notebook. 

!pip3 install librosa 

!pip3 install pydub 

Now, you have to import these libraries as follows:
<python>
import librosa 
import matplotlib.pyplot as plt
</python>
