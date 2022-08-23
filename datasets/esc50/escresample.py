import os 
from torchaudio.transforms import Resample
from torchaudio import load
from torch import save

escaudiopath = '/users2/local/tst-esc/ESC-50/audio/'
resampledpath = os.path.join(escaudiopath,'resampled')
os.makedirs(resampledpath,exist_ok=True)
transform = Resample(orig_freq=44100,new_freq=32000)

i=0
for curfile in os.listdir(escaudiopath):
    name,ext = os.path.splitext(curfile)
    if ext=='.wav':
        #print(f"Loading file {curfile}...")
        X,sr = load(os.path.join(escaudiopath,curfile))
        #print(f"Resampling file {curfile}...")
        Xresamp = transform(X)[0]
        filesave = os.path.join(resampledpath,f"{name}.pt")
        #print(f"Saving to file {filesave}")
        save(Xresamp,filesave)
        i+=1

print(f"Resampled a total of {i} files")