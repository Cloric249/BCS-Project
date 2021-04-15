import os
import json
import shutil
from tqdm import tqdm

# find the directory with the folders containing the images
directory = "D:/Downloads/frames_activitynet_5fps/activitynet_frames"
testing_ids = "D:/Downloads/captions/test_ids.json"
testing_folder = "D:/Downloads/testing"
folders_lost = 0
with open(testing_ids) as test_ids:
    ids = json.load(test_ids)

for each in tqdm(ids):
    if os.path.isdir(directory):
        for retry in range(100):
            try:
                shutil.move(
                    (directory + "/" + each), testing_folder)
                break
            except:
                if retry < 99:
                    print('rename failed, retrying...' )
                else:
                    folders_lost += 1
                    print('Folder not found')

print("Testing folders lost: ", folders_lost, "\n Percentage: ", folders_lost/len(ids))




