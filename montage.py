import glob
from moviepy.editor import VideoFileClip, clips_array
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog

# Counter
start = time.time()

# Define working directory
wdir = Path(os.getcwd())

# Create a tkinter root window (it won't be displayed)
root = tk.Tk()
root.withdraw()

# Open a dialog box to select participant's folder
idfolder = filedialog.askdirectory(initialdir=str(wdir))
id = os.path.basename(os.path.normpath(idfolder))

# Video pathways
processedvideos = idfolder + '/videos_processed/'

# Number of trials
trialfolders = sorted(glob.glob(processedvideos + '/*'))
ntrials = len(trialfolders)

for trial in trialfolders:

    # Trial name
    trialname = os.path.basename(trial)

    # Obtain raw videos
    rawvideos = sorted(glob.glob(processedvideos + trialname + '/*_refined.mp4'))
    ncams = len(rawvideos)

    # Compile videos (raw)
    allvideos = rawvideos.copy()
    vids = [VideoFileClip(video) for video in allvideos]

    # Combine videos together
    top_row = clips_array([vids[0:int(ncams/2)]])
    bot_row = clips_array([vids[int(ncams/2):]])
    final_video = clips_array([[top_row], [bot_row]])
    output_path = processedvideos + trialname + '/compilation.mp4'
    final_video.write_videofile(output_path, codec='libx264', fps=60)

# Counter
end = time.time()
print('Time to run code: ' + str(end - start) + ' seconds')