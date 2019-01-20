import moviepy.editor as mp
import os, imageio
import numpy as np

clips = []


# for i in range(180):
#     if(i%10==0):
#         a = imageio.imread("videos/butterfly/origin/frame"+str(i+1)+".png")
#         b = imageio.imread("videos/butterfly/gray/frame" + str(i + 1) + ".png")
#         c = imageio.imread("videos/butterfly/seq_result/frame" + str(i + 1) + ".png")
#     else:
#         a = imageio.imread("videos/butterfly/origin/frame" + str(i) + ".png")
#         b = imageio.imread("videos/butterfly/gray/frame" + str(i) + ".png")
#         c = imageio.imread("videos/butterfly/seq_result/frame" + str(i) + ".png")
#
#     abc = np.hstack([a, b, c])
#     imageio.imwrite("videos/butterfly/combine/"+str(i)+".png", abc)


filelist = os.listdir("videos/butterfly/combine")
filelist = [f for f in filelist if f.endswith(".png")]

for file in sorted(filelist, key=lambda x: int(x[:-4])):
    clips.append(mp.ImageClip("videos/butterfly/combine/"+file).set_duration(1/30))

concat_clip = mp.concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("videos/butterfly/combine.mp4", fps=30)