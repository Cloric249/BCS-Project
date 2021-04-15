import ast
import glob
import json
import os
import os.path
import pathlib
import cv2
import keyboard as KB
from Katna.video import Video
from PIL import Image
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm
import random
import shutil

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
    for folder in tqdm(os.listdir("D:/Downloads/validation")):
        FLAG = False
        img_array = []
        print(folder)
        for filename in glob.glob("D:/Downloads/validation/"+folder+"/*"):
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
            out = cv2.VideoWriter("D:/Downloads/validation_videos/"+folder+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
            verify = 0
            for i in range(len(img_array)):
                out.write(img_array[i])

            added.append(chkname)
            out.release()


def splitVideos():
    with open("val_2.json") as training:
        index = json.load(training)
    for video in tqdm(os.listdir("D:/Downloads/validation_videos")):
        try:
            vid = Video()
            vid_path = "D:/Downloads/validation_videos/" + video
            video = video[:-4]
            output_folder = "D:/Downloads/Seperated_validation_2/" + video + "/"
            out_dirpath = os.path.join(".", output_folder)
            print(out_dirpath)

            if not os.path.isdir(out_dirpath):
                pathlib.Path(out_dirpath).mkdir(parents=True, exist_ok=True)

            timestamps = index[video]["timestamps"]
            i = 0
            for x in timestamps:
                ffmpeg_extract_subclip(vid_path, x[0], x[1], targetname=out_dirpath + video + "__clip__" + str(i) + ".mp4")
                i = i + 1
        except:
            print("An error occured")


def extractKeyFrames():
    error = 0
    for folder in tqdm(os.listdir("D:/Downloads/Seperated_validation_1")):
        try:
            for video in os.listdir("D:/Downloads/Seperated_validation_1/" + folder):
                vid = Video()
                vid_path = "D:/Downloads/Seperated_validation_1/" + folder + "/" + video
                video = video[:-4]
                output_folder = "D:/Downloads/VKeyFrames_1/" + folder + "/" + video + "/"
                out_dirpath = os.path.join(".", output_folder)
                print(out_dirpath)

                if not os.path.isdir(out_dirpath):
                        pathlib.Path(out_dirpath).mkdir(parents=True, exist_ok=True)

                keyFrames = vid.extract_video_keyframes(1, file_path=vid_path)

                for counter, frame in enumerate(keyFrames):
                    vid.save_frame_to_disk(frame, file_path=out_dirpath, file_name="frame_"+str(counter), file_ext=".jpeg")
        except:
            error = error + 1

    print(error)

def cleanData():
    removed_files = open("removed files validation.txt", "w")
    removed = []
    for folder in os.listdir("D:/Downloads/VKeyFrames_1/"):
        if len(os.listdir("D:/Downloads/VKeyFrames_1/" + folder)) == 0:
            removed.append(folder)
            os.rmdir("D:/Downloads/VKeyFrames_1/" + folder)
        for each in os.listdir("D:/Downloads/VKeyFrames_1/" + folder):
            if len(os.listdir("D:/Downloads/VKeyFrames_1/" + folder + "/" + each)) == 0:
                removed.append(folder)
                os.rmdir("D:/Downloads/VKeyFrames_1/" + folder + "/" + each)


    removed_files.write(str(removed))
    removed_files.flush()
    removed_files.close()
    print("Number of folder removed: " + str(len(removed)))


def getFrames(id, clip_num, num_of_frames, mode):
    frames = []
    flag = None
    if mode == "train":
        if os.path.isdir("D:/Downloads/TKeyFrames/" + id + "/" + id + "__clip__" + str(clip_num)):
            for frame in tqdm(os.listdir("D:/Downloads/TKeyFrames/" + id + "/" + id + "__clip__" + str(clip_num))):
                try:
                    # Convert to 3 - channel
                    img = (Image.open("D:/Downloads/TKeyFrames/" + id + "/" + id + "__clip__" + str(clip_num) + "/" +
                                      frame)).convert("RGB")
                    frames.append(img)
                except:
                    print("Image was not found")
                    return None
        else:
            print("Directory not found")
            print(id)

    if mode == "validate":
        if os.path.isdir("D:/Downloads/VKeyFrames_1/" + id + "/" + id + "__clip__" + str(clip_num)):
            for frame in tqdm(os.listdir("D:/Downloads/VKeyFrames_1/" + id + "/" + id + "__clip__" + str(clip_num))):
                try:
                    # Convert to 3 - channel
                    img = (Image.open("D:/Downloads/VKeyFrames_1/" + id + "/" + id + "__clip__" + str(clip_num) + "/" +
                                      frame)).convert("RGB")
                    frames.append(img)
                except:
                    print("Image was not found")
                    return None
        else:
            print("Directory not found")
            print(id)

    return frames



if __name__ == '__main__':
    #framesToVid()
    #splitVideos()
    #extractKeyFrames()
    cleanData()
    print("Completed")
