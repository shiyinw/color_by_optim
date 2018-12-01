import cv2, imageio
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

  get_vedio_frames('videos/butterfly/butterfly.mp4', "videos/butterfly/origin/")

  for i in range(180):
    input_dir = input("input_dir : ")
    preprocessing.generate(input_dir="videos/butterfly/origin/frame{}.png".format(str(i)),
                           gray_dir="videos/butterfly/gray/frame{}.png".format(str(i)),
                           sketch_dir="videos/butterfly/sketch/frame{}.png".format(str(i)))