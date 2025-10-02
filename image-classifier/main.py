import os
from skimage.io import imread          # to load images
from skimage.transform import resize   # to resize images

from sklearn.model_selection import train_test_split, GridSearchCV  # data split + hyperparameter search
from sklearn.svm import SVC            # Support Vector Classifier
import numpy as np                     # numerical operations

# --- Prepare dataset paths ---
input_dir = os.path.join(os.path.dirname(__file__), '..', 'clf-data')  # go up one folder, then into 'clf-data'
input_path = os.path.abspath(input_dir)                                # get absolute path
categories = ['empty', "not_empty"]                                    # two labels (classes)

# --- Load and preprocess images ---
data = []     # list to hold image data
labels = []   # list to hold class labels
for category_idx, category in enumerate(categories):                   # loop over classes with index (0,1)
    for file in os.listdir(os.path.join(input_path, category)):        # loop over files inside each class folder
        image_path = os.path.join(input_path, category, file)          # build full path to file
        image = imread(image_path)                                     # read image
        image = resize(image, (15, 15))                                # resize to 15x15
        data.append(image.flatten())                                   # flatten 2D image into 1D vector
        labels.append(category_idx)                                    # store class index (0 or 1)

# --- Convert lists to numpy arrays ---
data = np.array(data)        # shape: (n_samples, n_features)
labels = np.array(labels)    # shape: (n_samples,)

# --- Split dataset into train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)  # 80% train, 20% test, balanced classes

# --- Define model ---
classifier = SVC()   # support vector classifier with default kernel (RBF)

# --- Define hyperparameter grid for tuning ---
paramers = {
    'gamma': [0.01, 0.001, 0.0001],   # influence of each training example
    'C': [1, 10, 100, 1000]           # margin hardness vs misclassification
}

# --- Search best parameters using cross-validation ---
grid_search = GridSearchCV(classifier, paramers, cv=5)   # 5-fold CV
grid_search.fit(X_train, y_train)                        # train model with parameter search

# --- Evaluate best model on test set ---
best_estimator = grid_search.best_estimator_             # model with best found params
accuracy = best_estimator.score(X_test, y_test)          # compute accuracy on unseen test data
print(f"Model accuracy: {accuracy * 100:.2f}%")          # print accuracy as percentage
