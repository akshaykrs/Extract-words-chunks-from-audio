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
```python
!pip3 install librosa 
!pip3 install pydub 
```
Now, you have to import these libraries as follows:

```python
import librosa 
import matplotlib.pyplot as plt
from pydub import AudioSegment
```
# Step 3: Feature Extractions: 

In this we will see four main features which every audio wav file have.

1. Zero Crossing: Rate at which the signal changes from positive to negative or back.
2. Spectral Centroids: Indicates where centre of mass of sound is located. 
3. Spectral RollOff: Frequency below which a specified percentage of total spectral energy. 
4. MFCC: (MOST IMPORTANT) A small set of features (10-20) which concisely describe the overall shape of a spectral envelope.

By the way we are not using this to make chunks but these are important to analyse audio and are must to learn before doing operations on audio files. 

# Step 4: Loading Audio: 

```python 
import librosa
audio_path = 'akw.wav'
x , sr = librosa.load(audio_path)
```
To check whether file is uploaded sucessfully plot as wwaveform as follows: 

```python 
%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
```
![alt text](https://github.com/akshaykrs/Extract-words-chunks-from-audio/blob/master/fig.%201.png "Fig. 1")
