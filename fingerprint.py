
import librosa, numpy as np
from scipy.ndimage import maximum_filter

def load_audio(path_or_file):
    return librosa.load(path_or_file, sr=None, mono=True)

def spectrogram(audio):
    S = np.abs(librosa.stft(audio,n_fft=2048,hop_length=512))
    return S

def peaks(S, percentile=99.5):
    th=np.percentile(S,percentile)
    lm=maximum_filter(S,size=20)
    p=np.where((S==lm)&(S>th))
    return list(zip(p[0],p[1]))

def hashes(pk, fan=10):
    pk=sorted(pk,key=lambda x:x[1])
    out=[]
    for i in range(len(pk)):
        for j in range(i+1,min(i+fan,len(pk))):
            f1,t1=pk[i]; f2,t2=pk[j]
            if t2>t1:
                out.append(((f1,f2,t2-t1),t1))
    return out
