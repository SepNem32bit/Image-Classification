import os
import pickle
import numpy as np

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Prepare data
input_dir='D:\Data Science\Python Assignment\Computer Vision\Data\clf-data'
categories=['empty','not_empty']

data=[]
labels=[]
for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path=os.path.join(input_dir,category,file)
        img=imread(img_path)
        img=resize(img,(15,15),mode='constant', anti_aliasing=True, preserve_range=True)
        # print(img.shape)
        # print(img.flatten().shape)
        #One flat array
        data.append(img.flatten())
        labels.append(category_idx)

data=np.asarray(data,dtype=object)
labels=np.asarray(labels,dtype=object)

# train / test split

#strtify is used to keep the same distribution of labels in train and test

#It's neccessary to transform the label type to integer
x_train, x_test, y_train, y_test= train_test_split(data, labels.astype(int), test_size=0.2,shuffle=True, stratify=labels)



# train classifier
classifier = SVC()

# C is a hyperparameter in SVM to control error
# Gamma decides how much curvature we want in a decision boundary.
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# Print dimensions and types of data before grid search
print("Dimensions of x_train:", x_train.shape)
print("Dimensions of y_train:", y_train.shape)
print("Types of x_train and y_train:", type(x_train), type(y_train))


grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)


#test performance
best_estimator=grid_search.best_estimator_

y_prediction=best_estimator.predict(x_test)
score=accuracy_score(y_prediction,y_test)
print('{} is our accuracy'.format(str(score)))

#Saving the model
pickle.dump(best_estimator,open('D:\Data Science\Python Assignment\Computer Vision\Model\CarParkImageClassification.p','wb'))