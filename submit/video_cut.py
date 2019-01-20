import cv2, imageio, os
import preprocessing
import matplotlib.pyplot as plt


dir = "videos/bird/"


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

  if (not os.path.exists("{}origin".format(dir))):
    os.mkdir("{}origin".format(dir))
    get_vedio_frames('videos/bird/bird.mp4', "videos/bird/origin/")
  if (not os.path.exists("{}gray".format(dir))):
    os.mkdir("{}gray".format(dir))
  if (not os.path.exists("{}sketch".format(dir))):
    os.mkdir("{}sketch".format(dir))


  for i in range(138):
    preprocessing.generate(input_dir="videos/bird/origin/frame{}.png".format(str(i)),
                           gray_dir="videos/bird/gray/frame{}.png".format(str(i)),
                           sketch_dir="videos/bird/sketch/frame{}.png".format(str(i)),
                           small=100, big=500)
    print(i)