
# coding: utf-8

# # Extracting Words From Audio File (.Wav) without any srt.

# Here we are going to extract words and made separate chunks for every words from input audio file i.e. .wav file. If you have mp4 or mp3 files no worries I will provide command to convert them also. So, lets get started... 

# I have audio after noise reduction and other preprocessing. You can find these. 

# <h2>Importing Libraries</h2>

# In[4]:


import librosa 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display


# <h2> Converting from mp3 or mp4 to .wav (audio file)</h2>

# Use these commands to convert and for this you should have ffmpeg installed 

# <b>1. command2mp3 = "ffmpeg -i (your_audio).mp4 (new_audio).mp3"</b> // if you have mp4 file follow both command. 
# 
# 
# <b>2. command2wav = "ffmpeg -i (new_audio).mp3 (needed_audio).wav"</b>  // if you have mp3 file use only this command.

# <h2>Loading Audio File</h2>

# In[5]:


audio_path = 'akw.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))


# In[6]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# <h2>identifying vaious sounds</h2>

# In[18]:


from pydub import AudioSegment

beats = []

tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time',trim=True)

print(len(beat_times))
beats = beat_times

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r')
plt.ylim(-1, 1)    


# <h2>Making Chunks as identified above</h2> 

# In[24]:


start_time = 0
end_time = 0
count = 0

for i in beat_times:
    if i != max(beat_times):
        start = beat_times[count]*1000
        end = beat_times[count+1]*1000
        newAudio = AudioSegment.from_wav("akw.wav")
        newAudio = newAudio[start:end]
        newAudio.export(('newSong{}.wav').format(count), format="wav")
        print('Chunk {} is build.'.format(count))
    else:
        newAudio = AudioSegment.from_wav("akww.wav")
        start = max(beat_times)*1000
        newAudio = newAudio[start:]
        newAudio.export('new.wav', format="wav")
        print('Chunk {} is build.'.format(count))
    count = count + 1
    


# <h2> Spectrogram Repersentation </h2>

# In[7]:


#display Spectrogram
X = librosa.stft(x) #short term fourier transform 
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb,sr=sr, x_axis='time', y_axis='hz') 


#If to print log of frequencies  
librosa.display.specshow(Xdb,sr=sr,  x_axis='time', y_axis='log')
plt.colorbar()


# <h2> Features Extractions </h2>

# In this we will see four main features which every audio wav file have.
# 
# 1. Zero Crossing: Rate at which the signal changes from positive to negative or back.
# 2. Spectral Centroids: Indicates where centre of mass of sound is located.
# 3. Spectral RollOff: Frequency below which a specified percentage of total spectral energy.
# 4. MFCC: (MOST IMPORTANT) A small set of features (10-20) which concisely describe the overall shape of a spectral envelope.

# <h3> Zero Crossing </h3>

# In[10]:


n0 = 0
n1 = 800000 #I am taking for 800 seconds
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))


# In[11]:


# we can also have zero crossing rate 

librosa.feature.zero_crossing_rate(x)


# <h2>spectral_centroids</h2>

# In[12]:


#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')


# <h2>spectral_rolloff</h2>

# In[14]:


spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')


# <h2>mfccs</h2>

# In[15]:


mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

