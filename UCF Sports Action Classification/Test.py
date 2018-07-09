import cv2
vidcap = cv2.VideoCapture('/Users/abhianshusingla/Downloads/ucf_sports_actions/ucf action/Walk-Front/001/3206-12_70000.avi')
count = 0
success = True
while success:
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1
print(count)
