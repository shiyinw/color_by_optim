import cv2, os, imageio
from moviepy.video.io.VideoFileClip import VideoFileClip
import preprocessing

def get_vedio_frames(video_dir, output_dir):
  vidcap = cv2.VideoCapture(video_dir)
  success, image = vidcap.read()
  count = 0
  while success:
    image = cv2.resize(image, (640, 480))
    imageio.imwrite("{}frame{}.png".format(output_dir, count), image)  # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
  vidcap.release()
  cv2.destroyAllWindows()



if __name__ == "__main__":

  # if(not os.path.exists("videos/mickey/gray.mp4")):
  #   with VideoFileClip("videos/mickey/origin.mp4") as video:
  #     new = video.subclip(393, 398.5)
  #     new.write_videofile("videos/mickey/gray.mp4", audio_codec='aac')


  #get_vedio_frames('videos/butterfly/butterfly.mp4', "videos/butterfly/origin/")

  for i in range(180):
    print(i)
    preprocessing.generate(input_dir="videos/butterfly/origin/frame{}.png".format(str(i)),
                           gray_dir="videos/butterfly/gray/frame{}.png".format(str(i)),
                           sketch_dir="videos/butterfly/sketch/frame{}.png".format(str(i)))