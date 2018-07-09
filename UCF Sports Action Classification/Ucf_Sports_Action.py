import os
from PIL import Image
from pylab import *
from scipy import *
#from Histogram_of_Gradients import HoG
from skimage.feature import hog
from skimage import data, exposure

featureList = []
labelList = []
frames = []
frames.append(0)

# Features extracted are:
# Global Features -
# Active Contour Models
# Local Features -
# 1. Histogram
# 2. Spatial Features
# 3. Histogram of Gradients - RGB
def features(img):
    I = array(img)
    feature = []

    # Histogram of Gradients feature
    # Orientations = 9
    for channel in range(I.shape[2]):
        fd, hog_image = hog(I[:,:,channel], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True, transform_sqrt=True, feature_vector=True)
        feature.extend(fd)

    # Histogram feature
    for channel in range(I.shape[2]):
        hg = np.histogram(I[:,:,channel],bins = 16)
        # Taking only the number of pixels, discarding the range of intensity
        feature.extend(hg[0])

    # Spatial Features
    sf = array(img.resize((32,32))).ravel()
    feature.extend(sf)

    return feature


# Reading files
in_path = "/Users/abhianshusingla/Downloads/ucf_sports_actions/ucf action/"
in_length = len(in_path)
labelsMap = {"diving-side" :  "1", "golf-swing-back" : "2", "golf-swing-front" : "3", "golf-swing-side" : "4", "kicking-front" : "5", "kicking-side" : "6", "lifting" : "7", "riding-horse" : "8",
 "run-side" : "9", "skateboarding-front" : "10", "swing-bench" : "11", "swing-sideangle" : "12", "walk-front" : "13"}

# labelsMap = {"diving-side" :  "1", "golf-swing" : "2", "kicking" : "3", "lifting" : "4", "riding-horse" : "5",
#  "run-side" : "6", "skateboarding-front" : "7", "swing-bench" : "8", "swing-sideangle" : "9", "walk-front" : "10"}

for folder_name in os.listdir(in_path):

    # action names
    if(folder_name[0] != '.'):
        in_path += folder_name

        test_case = 0

        # action numbers
        for sub_folder in os.listdir(in_path):
            print(folder_name, sub_folder)
            count = 0

            if(sub_folder[0] != '.'):

                in_sublength = len(in_path)
                in_path += "/" + sub_folder

                # file names
                for file_name in os.listdir(in_path):

                    if(file_name == 'jpeg'):
                        for file_names in os.listdir(in_path + "/" + file_name):

                            file_path = in_path + "/jpeg/" + file_names

                            if(file_path.endswith('.jpg')):
                                file_gt = in_path + "/gt/" + os.path.splitext(file_names)[0] +'.tif.txt'

                                if(os.path.exists(file_gt)):

                                    gt_file = open(file_gt, "r")
                                    cropped_area = (gt_file.read()).split()

                                    img = Image.open(file_path).resize((64,128))
                                    feature = features(img)

                                    featureList.append(feature)
                                    labelList.append(labelsMap[folder_name.lower()])
                                    count += 1


                    elif(file_name[0] != '.'):

                        in_file = in_path + "/" + file_name
                        file_gt = in_path + "/gt/" + os.path.splitext(file_name)[0] +'.tif.txt'

                        # Lifting action doesn't have gt files
                        if(os.path.exists(in_file) and os.path.exists(file_gt) and in_file.endswith('.jpg')):

                            gt_file = open(file_gt, "r")
                            cropped_area = (gt_file.read()).split()
                            # Preprocessing the images
                            # print(int(cropped_area[0]),int(cropped_area[1]),int(cropped_area[0])+int(cropped_area[2]),int(cropped_area[1])+int(cropped_area[3]),cropped_area[4])
                            crop_coord = (int(cropped_area[0]),int(cropped_area[1]),int(cropped_area[0])+int(cropped_area[2]),int(cropped_area[1])+int(cropped_area[3]))

                            img = Image.open(in_file)
                            img = img.crop(crop_coord).resize((64,128))

                            feature = features(img)

                            featureList.append(feature)
                            labelList.append(labelsMap[folder_name.lower()])
                            count += 1

                frames.append(frames[len(frames) - 1] + count)

                in_path = in_path[:in_sublength]

        in_path = in_path[:in_length]

print(len(featureList))
print(len(labelList))
print(len(frames))
print(frames)

# feature vector 53-58 (lifting - no gt), 147-149(Walking - no gt)
