import os

out_path = "/Users/abhianshusingla/Downloads/output/ucf_sports_actions/ucf action/"
out_length = len(out_path)
feature_path = out_path + "feature_vector"
label_path = out_path + "labels"
cycles = 150
test_case = 0

for i in range(1,cycles+1):

    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    average_accuracy = 0

    for files in os.listdir(out_path):

        if(files[0:14] == "feature_vector" and files[14:] != str(i) + ".txt"):
            #print("Train ", out_path+files)
            with open(out_path+files) as f:
                lines = f.readlines()
                for each_line in lines:
                    values = each_line.split(' ')
                    vectors = []
                    for element in values:
                        vectors.append(float(element))
                    features_train.append(vectors)
        elif(files[0:14] == "feature_vector" and files[14:] == str(i) + ".txt"):
            #print("Test ", out_path+files)
            with open(out_path+files) as f:
                lines = f.readlines()
                for each_line in lines:
                    values = each_line.split(' ')
                    vectors = []
                    for element in values:
                        vectors.append(float(element))
                    features_test.append(vectors)

        if(files[0:6] == "labels" and files[6:] != str(i) + ".txt"):
            #print("Train ", out_path+files)
            with open(out_path+files) as f:
                lines = f.readlines()
                for each_label in lines:
                    labels_train.append(float(each_label))
        elif(files[0:6] == "labels" and files[6:] == str(i) + ".txt"):
            #print("Test ", out_path+files)
            with open(out_path+files) as f:
                lines = f.readlines()
                for each_label in lines:
                    labels_test.append(float(each_label))

    from sklearn import svm
    clf = svm.SVC()
    clf.fit(features_train,labels_train)

    print(i, "Trained")
    y_pred = clf.predict(features_test)
    print(y_pred)
    accuracy = clf.score(features_test,labels_test)
    average_accuracy += accuracy
    print(i, accuracy)

print(average_accuracy/150)





'''
X = []
Y = []
with open('outfile1.txt') as f:
    lines = f.readlines()
    for each_line in lines:
        values = each_line.split(' ')
        vectors = []
        for element in values:
            vectors.append(float(element))
        X.append(vectors)


with open('outfile2.txt') as f:
    lines = f.readlines()
    for each_label in lines:
        Y.append(float(each_label))

from sklearn import svm
clf = svm.SVC()
clf.fit(X,Y)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)


# y_ = clf.predict(..)
# print(y_)
'''
