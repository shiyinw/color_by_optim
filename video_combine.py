import moviepy.editor as mp
import os

clips = []

filelist = os.listdir("videos/butterfly/seq_result")
filelist = [f for f in filelist if f.endswith(".png") and not f.endswith("0.png")]

for file in sorted(filelist, key=lambda x: int(x[5:-4])):
    if(file.endswith(".png")):
        clips.append(mp.ImageClip("videos/butterfly/seq_result/"+file).set_duration(1/30))

concat_clip = mp.concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("videos/butterfly/result_1129.mp4", fps=24)