
# coding: utf-8

# ### Importing required Libraries

# In[ ]:


import cv2
import numpy as np 
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import glob
from ComputeSIFT import DetectandCompute


# ### Reading food image file names for testing and training purpose

# In[20]:


def readFileName():
    # Image path for food-101 dataset
    food_items = glob.glob('/Users/abhianshusingla/Downloads/food-101/images/*')
    
    # Curently doing the testing for only 20 food classes
    food_items = food_items[:20]
    
    # Split the image file name food into food_train and food_test 
    food_train = []
    food_test = []

    for f in food_items:
        food = glob.glob(f + '/*.jpg')
        
        # Taking first 200 images for training purpose
        food_train.append(food[:200])
        
        # Taking 20 images for each class for testing purpose
        food_test.append(food[200:220])
        
    return food_train, food_test


# ### Extracting the SIFT features for both training and testing images

# In[21]:


def extractSIFT(food):
    
    # Count of the number of keypoints for a particular image
    count_keypoints = []
    
    # List of image descriptors
    # Size equals to the number of images
    food_descriptors = []
    
    # It is a vertical stack of all the images descriptors, so if the total number of descriptors 
    # for the combined images is 10000, then vertical_stack length is 10000 and each element will have 128 elements
    vertical_stack = []
    
    # Label names of the food data
    labels = []
    
    # food contains the list of the food class name
    for i in range(len(food)):
        
        # food[i] contains the list of images for that particular food class food[i]
        for j in range(len(food[i])):
            
            # keypoints and descriptors calculated using SIFT
            _, des = DetectandCompute(food[i][j])
            
            # Add the descriptors to the list of food_descriptors
            food_descriptors.append(des)
            
            # make a vertical stack of the elements having length 128( feature vector length)
            vertical_stack.extend(des)
            
            # Store the count of number of keypoints detected for a particular image
            # Needed while calculating the histogram
            count_keypoints.append(des.shape[0])
            
            # Assigning the label names according to the folder name
            label_name = food[i][j].split('/')
            label_name = label_name[-2]
            labels.append(label_name)
            
    return count_keypoints, food_descriptors, vertical_stack, labels   


# ### Normalization

# In[22]:


# Normalizing the data so that large values fit within a small range, it will decrease the time in fitting the data for a classifier
from sklearn.preprocessing import StandardScaler
def normalize(vertical_stack):
    X = np.array(vertical_stack).astype(np.float64)
    
    # StandardScaler function
    X_scaler = StandardScaler().fit(X)
    
    # Transforming the data within small range
    vertical_stack = X_scaler.transform(X)
    
    return vertical_stack.tolist()


# ### Clustering

# In[23]:


def clustering(vertical_stack, clusters_count):
    
    # K Means Cluster classifier 
    cluster_clf = KMeans(n_clusters = clusters_count)
    
    # Predicting the cluster for every keypoint 
    pred_clusters = cluster_clf.fit_predict(vertical_stack)
    
    return pred_clusters


# ### Vocabulary - Histogram of Clusters

# In[24]:


def histogram_clusters(count_keypoints,pred_clusters,clusters_count):
    
    # List of Histogram of all the images
    # Histogram obtained is train_X or test_X
    histogram = []
    count = 0
    
    # count_keypoints is the list of the number of keypoints for an image
    for i in range(len(count_keypoints)):
        
        # Make an array equal to the size of the histogram
        hist = np.zeros(clusters_count)
        
        # Create histogram of clusters 
        for j in range(count_keypoints[i]):
            hist[pred_clusters[count + j]] += 1
        
        count += count_keypoints[i]
        
        # Append the calculated hist of the image to histogram list
        histogram.append(hist)
    return histogram


# ### Function used for extracting the features and assigning those to train_X, train_Y, test_X and test_Y 

# In[25]:


def extractData():
    food_train, food_test = readFileName()
    
    print("SIFT..")
    print("SIFT of train data..")
    count_keypoints_train, food_descriptors_train, vertical_stack_train, train_Y = extractSIFT(food_train)
    print("SIFT of test data..")
    count_keypoints_test, food_descriptors_test, vertical_stack_test, test_Y = extractSIFT(food_test)
    
    print("Normalization..")
    print("Normalization of train data..")  
    vertical_stack_train = normalize(vertical_stack_train)
    print("Normalization of test data..")  
    vertical_stack_test = normalize(vertical_stack_test)
    
    print("Clustering..")
    clusters_count = 100
    print("Clustering of train data..")
    pred_clusters_train = clustering(vertical_stack_train, clusters_count)
    print("Clustering of test data..")
    pred_clusters_test = clustering(vertical_stack_test, clusters_count)
    
    print("Histogram..")
    print("Histogram of train data..")
    train_X = histogram_clusters(count_keypoints_train,pred_clusters_train,clusters_count)
    print("Histogram of test data..")
    test_X = histogram_clusters(count_keypoints_test,pred_clusters_test,clusters_count)
    
    return train_X, train_Y, test_X, test_Y


# In[26]:


train_X, train_Y, test_X, test_Y = extractData()
print("Data is Extracted for training and testing")


# ### Support Vector Classifier

# In[33]:


# Support Vector Machine - Classifier
from sklearn import svm
from sklearn.svm import SVC

def SVC_Classifier(train_X, train_Y, test_X, test_Y):
    # Classifier parameters
    # kernel = "rbf" or "linear"
    # If gamma is ‘auto’ then 1/n_features will be used.
    clf = SVC(C = 1.0, kernel = "rbf", gamma = "auto")

    # Fit the classifier with features_train and labels_train
    clf.fit(train_X,train_Y)

    # Predicting the label values
    y_pred = clf.predict(test_X)

    # Accuracy
    accuracy = clf.score(test_X,test_Y)
    
    return accuracy


# In[34]:


accuracy = SVC_Classifier(train_X, train_Y, test_X, test_Y)
print("Accuracy of the food data with SVM is ",accuracy)


# ### Random Forest Classifier

# In[30]:


from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

def Forest_Classifier(train_X, train_Y, test_X, test_Y):

    # If max_depth value is high, then it won't segregate the data clearly into leaf nodes,
    # but if the max_depth value is very low, then it it will take long time to train the data
    clf = RandomForestClassifier(max_depth = 8)

    # Fit the classifier with features_train and labels_train
    clf.fit(train_X,train_Y)

    # Predicting the label values
    pred_Y = clf.predict(test_X)

    # Accuracy
    accuracy = clf.score(test_X,test_Y)

    return accuracy


# In[31]:


accuracy = Forest_Classifier(train_X, train_Y, test_X, test_Y)
print("Accuracy of the food data with Forest Classifier is ",accuracy)

