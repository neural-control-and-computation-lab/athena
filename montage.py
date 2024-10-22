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
rawvideos = idfolder + '/videos/'
processedvideos = idfolder + '/videos_processed/'

# Number of trials
trialfolders = sorted(glob.glob(processedvideos + '/*'))
ntrials = len(trialfolders)

# Use raw or 2D predictions overlaid raw?
useraw = False

for trial in trialfolders:

    # Trial name
    trialname = os.path.basename(trial)

    # Obtain raw videos
    if useraw is True:
        vidlist = sorted(glob.glob(rawvideos + trialname + '/*.avi'))
    else:
        vidlist = sorted(glob.glob(processedvideos + trialname + '/*_refined.mp4'))
    ncams = len(vidlist)

    # Obtain 3D landmark
    video3d = processedvideos + trialname + '/data3d.mp4'

    # Compile videos (raw + 3D landmarks)
    allvideos = vidlist.copy()
    allvideos.append(video3d)
    vids = [VideoFileClip(video) for video in allvideos]

    # Resize 3D data to appear bigger
    vids[-1] = vids[-1].resize(1.5)

    # Combine videos together
    top_row = clips_array([vids[0:4]])
    mid_row = clips_array([vids[4:-1]])
    bot_row = clips_array([[vids[-1]]])
    final_video = clips_array([[top_row], [mid_row], [bot_row]])
    output_path = processedvideos + trialname + '/compilation.mp4'
    final_video.write_videofile(output_path, codec='libx264', fps=60)

# Counter
end = time.time()
print('Time to run code: ' + str(end - start) + ' seconds')
