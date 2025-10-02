import os

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.svm import SVC

import numpy as np

#prpare data
input_dir = os.path.join(os.path.dirname(__file__), '..', 'clf-data')
input_path = os.path.abspath(input_dir)
categories = ['empty', "not_empty"]

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_path, category)):
        image_path = os.path.join(input_path, category, file)
        image = imread(image_path)
        image = resize(image, (15, 15))
        data.append(image.flatten()) 
        labels.append(category_idx)


data = np.array(data)
labels = np.array(labels)


#train / test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#train model
classifier = SVC()

paramers = {'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100, 1000]}

grid_search = GridSearchCV(classifier, paramers, cv=5)
grid_search.fit(X_train, y_train)

#test performance
best_estimator = grid_search.best_estimator_
accuracy = best_estimator.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
