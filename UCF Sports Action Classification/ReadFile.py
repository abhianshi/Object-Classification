import os
from PIL import Image
from pylab import *
from scipy import *
#from Histogram_of_Gradients import HoG
from skimage.feature import hog
from skimage import data, exposure

def features(img):
    I = array(img)
    feature = []

    # Histogram of Gradients feature
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
out_path = "/Users/abhianshusingla/Downloads/output/ucf_sports_actions/ucf action/"
in_length = len(in_path)
out_length = len(out_path)
labelsMap = {"diving-side" :  "1", "golf-swing-back" : "2", "golf-swing-front" : "3", "golf-swing-side" : "4", "kicking-front" : "5", "kicking-side" : "6", "lifting" : "7", "riding-horse" : "8",
 "run-side" : "9", "skateboarding-front" : "10", "swing-bench" : "11", "swing-sideangle" : "12", "walk-front" : "13"}

# labelsMap = {"diving-side" :  "1", "golf-swing" : "2", "kicking" : "3", "lifting" : "4", "riding-horse" : "5",
#  "run-side" : "6", "skateboarding-front" : "7", "swing-bench" : "8", "swing-sideangle" : "9", "walk-front" : "10"}


count = 1

for folder_name in os.listdir(in_path):

    # action names
    if(folder_name[0] != '.'):
        in_path += folder_name
        out_path += folder_name

        test_case = 0
        # action numbers
        for sub_folder in os.listdir(in_path):

            if(sub_folder[0] != '.'):

                featureList = []
                labelList = []

                in_sublength = len(in_path)
                out_sublength = len(out_path)
                in_path += "/" + sub_folder
                out_path += "/" + sub_folder

                # file names
                for file_name in os.listdir(in_path):

                    try:
                        os.makedirs(os.path.dirname(out_path + "/" + file_name))
                    except OSError as e:
                        if e.errno != os.errno.EEXIST:
                            raise

                    if(file_name == 'jpeg'):
                        for file_names in os.listdir(in_path + "/" + file_name):

                            file_path = in_path + "/jpeg/" + file_names
                            out_file = out_path + "/" + file_names

                            if(file_path.endswith('.jpg')):
                                Image.open(file_path).resize((64,128)).save(out_file)

                                file_gt = in_path + "/gt/" + os.path.splitext(file_names)[0] +'.tif.txt'

                                if(os.path.exists(file_gt)):

                                    gt_file = open(file_gt, "r")
                                    cropped_area = (gt_file.read()).split()

                                    img = Image.open(out_file)

                                    feature = features(img)

                                    featureList.append(feature)
                                    labelList.append(labelsMap[folder_name.lower()])


                    elif(file_name[0] != '.'):

                        in_file = in_path + "/" + file_name
                        file_gt = in_path + "/gt/" + os.path.splitext(file_name)[0] +'.tif.txt'
                        out_file = out_path + "/" + file_name

                        # Lifting action doesn't have gt files
                        if(os.path.exists(in_file) and os.path.exists(file_gt) and in_file.endswith('.jpg')):

                            img = Image.open(in_file)
                            gt_file = open(file_gt, "r")
                            cropped_area = (gt_file.read()).split()
                            # Preprocessing the images
                            # print(int(cropped_area[0]),int(cropped_area[1]),int(cropped_area[0])+int(cropped_area[2]),int(cropped_area[1])+int(cropped_area[3]),cropped_area[4])
                            crop_coord = (int(cropped_area[0]),int(cropped_area[1]),int(cropped_area[0])+int(cropped_area[2]),int(cropped_area[1])+int(cropped_area[3]))
                            try:
                                img.crop(crop_coord).resize((64,128)).save(out_file)
                                img = Image.open(out_file)

                                feature = features(img)

                                featureList.append(feature)
                                labelList.append(labelsMap[folder_name.lower()])

                            except OSError as e:
                                if e.errno != os.errno.EEXIST:
                                    raise

                features_list = out_path[:out_length] + "/feature_vector" + str(count) +".txt"
                labels_list = out_path[:out_length] + "/labels" + str(count) + ".txt"

                with open(features_list,"w") as f:
                    f.write("\n".join(" ".join(map(str, a)) for a in featureList))

                with open(labels_list,"w") as f:
                    f.write("\n".join(labelList))

                count += 1

                in_path = in_path[:in_sublength]
                out_path = out_path[:out_sublength]

        in_path = in_path[:in_length]
        out_path = out_path[:out_length]


# feature vector 53-58 (lifting - no gt), 147-149(Walking - no gt)


f.close()
