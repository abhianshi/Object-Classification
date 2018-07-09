import os
from PIL import Image
import math
import numpy as np
from pylab import *
from scipy import *

# X-Derivative - 1D mask
def XDerivative(I):
    mask = [-1, 0, 1]
    Ix = np.zeros(I.shape)
    row,column = I.shape
    for i in range(row):
        for j in range(1,column-1):
            Ix[i][j] = I[i][j-1] * mask[0] + I[i][j] * mask[1] + I[i][j+1] * mask[2]
    return Ix

# Y-Derivative - 1D mask
def YDerivative(I):
    mask = [-1, 0, 1]
    Iy = np.zeros(I.shape)
    row,column = I.shape
    for i in range(1,row-1):
        for j in range(column):
            Iy[i][j] = I[i-1][j] * mask[0] + I[i][j] * mask[1] + I[i+1][j] * mask[2]
    return Iy

# Magnitude of gradient
def magnitude(Ix,Iy):
    mag = np.zeros(Ix.shape)
    row,column = Ix.shape
    for i in range(row):
        for j in range(column):
            mag[i][j] = math.sqrt(Ix[i][j] ** 2 + Iy[i][j] ** 2)
            #print(mag[i][j],end = ' ')
    return mag

# Direction of gradients
def direction(Ix,Iy):
    angle = np.zeros(Ix.shape)
    row,column = Ix.shape
    for i in range(row):
        for j in range(column):

            if (Ix[i][j] == 0):
                angle[i][j] = int(math.degrees(math.pi/2))
            else:
                theta = int(math.degrees(math.atan(Iy[i][j]/Ix[i][j])))
                if(theta < 0):
                    angle[i][j] = 180 + theta
                else:
                    angle[i][j] = theta
        #     print(angle[i][j],end = ' ')
        # print()

    return angle


def cell_8(mag,angle):
    cell = np.zeros((9))

    for i in range(8):
        for j in range(8):
            div = int(angle[i][j] / 20)
            mod = int(angle[i][j] % 20)
            cell[(div+1)%9] += (mod/20.0) * (mag[i][j])
            cell[div] += ((20 - mod)/20.0) * (mag[i][j])
            #print(mag[i][j],angle[i][j],cell[(div+1)%9],cell[div])
    return cell


def normalized_cell(myList):
    normalized_list = []

    # Size of myList is 8 * 16 as image size is 64 * 128
    for i in range(7 * 15):
        a = np.zeros((36))
        a[0:9] = myList[i]
        a[9:18] = myList[i+1]
        a[18:27] = myList[i+8]
        a[27:36] = myList[i+8+1]

        normalized_list.append(a)

    return normalized_list

def featureVector(normalized_list):

    feature_vector = np.zeros((36*7*15))
    for i in range(7 * 15):
        feature_vector[36*i:36*i+36] = normalized_list[i]

    return feature_vector


# Histogram of Gradients
def HoG(I):

    # 2. Calculate Gradients
    Ix = XDerivative(I)
    Iy = YDerivative(I)
    mag = magnitude(Ix,Iy)
    angle = direction(Ix,Iy)

    # Image is 64 * 128, so dividing into (8,16) 8*8 cells
    myList = []
    row,column = mag.shape

    # 3. Calculate Histogram of Gradients in 8 * 8 cells
    for i in range(int(row/8)):
        for j in range(int(column/8)):
            myList.append(cell_8(mag[8*i:8*i+8, 8*j:8*j+8], angle[8*i:8*i+8, 8*j:8*j+8]))

    # 4. Block Normalization
    normalized_list = normalized_cell(myList)

    # 5. Calculate Feature Vector
    feature_vector = featureVector(normalized_list)

    return feature_vector
