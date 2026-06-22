
import os,pickle
from fingerprint import *

db={}
for song in os.listdir("songs"):
    if song.lower().endswith((".mp3",".wav")):
        a,sr=load_audio(os.path.join("songs",song))
        h=hashes(peaks(spectrogram(a)))
        for hh,t in h:
            db.setdefault(hh,[]).append((song,t))

pickle.dump(db,open("database.pkl","wb"))
print("Songs indexed:",len([x for x in os.listdir('songs') if x.endswith(('.mp3','.wav'))]))
print("Hashes:",len(db))
