import torch
import numpy as np
import torchvision
import torchaudio
import cv2
import keyboard as KB
import glob
import os
import os.path
from Katna.video import Video
from tqdm import tqdm
import pathlib
from PIL import Image


# read the video and return the frames
def test():
    video_path = "videos/Best Tennis Point Ever - Under 8 UK Kent Mini Red Tennis Championship 2010.mp4"


    capture = cv2.VideoCapture(video_path)
    # number of frames of video
    num_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_of_frames)
    # Check if video stream opened successfully
    if (capture.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (capture.isOpened()):
        # Capture frame-by-frame
        ret, frame = capture.read()
        print(frame)
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
        # wait one seconde before moving onto next frame
        cv2.waitKey(1000)
        if KB.is_pressed('q'):
            print("Ending Sequence...")
            capture.release()

    # when finished, release the video stream from memory
    capture.release()


# TODO: Implement an agent capable of assessing which frames provide the most information

# TODO: Implement algorithm that decides how many frames should be taken from the video

def framesToVid():
    added = []
    for folder in tqdm(os.listdir("D:/Downloads/training")):
        FLAG = False
        img_array = []
        print(folder)
        for filename in glob.glob("D:/Downloads/training/"+folder+"/*"):
            chkname = folder + ".avi"
            if chkname not in added:
                try:
                    img = cv2.imread(filename)
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)
                except:
                    print("Corrupted Image")
            else:
                    FLAG = True

        if FLAG == False:
            out = cv2.VideoWriter("D:/Downloads/training_videos/"+folder+".avi", cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            added.append(chkname)
            out.release()

def extractKeyFrames():
    for video in tqdm(os.listdir("D:/Downloads/training_videos")):
        try:
            vid = Video()
            vid_path = "D:/Downloads/training_videos/" + video
            video = video[:-4]
            output_folder = "D:/Downloads/TKeyFrames/" + video + "/"
            out_dirpath = os.path.join(".", output_folder)
            print(out_dirpath)

            if not os.path.isdir(out_dirpath):
                pathlib.Path(out_dirpath).mkdir(parents=True, exist_ok=True)


            keyFrames = vid.extract_frames_as_images(20, file_path=vid_path)

            for counter, frame in enumerate(keyFrames):
                vid.save_frame_to_disk(frame, file_path=out_dirpath, file_name="frame_"+str(counter), file_ext=".jpeg")
        except:
            print("An error occured")

def getFrames(id, num_of_frames):
    frames = []
    flag = None
    for frame in tqdm(os.listdir("D:/Downloads/TKeyFrames/" + id)):
        try:
            # Convert to 3 - channel
            img = (Image.open(frame)).convert('RGB')
            frames.append(img)
        except:
            print("An error occured")
    removing = len(frames) > num_of_frames
    while removing != 0:
        if flag == False:
            # remove the image at the quarter position
            img_index = len((frames//2)//2)
            frames.pop(img_index)
            flag = True
        else:
            img_index = len((frames//2)//2)
            frames.pop(img_index)
            flag = False

    return frames



if __name__ == '__main__':
    #framesToVid()
    #extractKeyFrames()
    print("OK")
