import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images, labels = [], []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (255,255))  # Resizing to a fixed shape
            images.append(img_resized)
            labels.append(label)

    return np.array(images), np.array(labels)

# Paths for training and testing folders
train_infected_path = '/kaggle/input/pcos-detection-using-ultrasound-images/data/train/infected'
train_non_infected_path = '/kaggle/input/pcos-detection-using-ultrasound-images/data/train/notinfected'
test_infected_path = '/kaggle/input/pcos-detection-using-ultrasound-images/data/test/infected'
test_non_infected_path = '/kaggle/input/pcos-detection-using-ultrasound-images/data/test/notinfected'

# Load training data
train_infected_images, train_infected_labels = load_images_from_folder(train_infected_path, 1)
train_non_infected_images, train_non_infected_labels = load_images_from_folder(train_non_infected_path, 0)

# Ensure both classes have the same shape before stacking
if train_infected_images.shape[1:] != train_non_infected_images.shape[1:]:
    raise ValueError("Error: Image dimensions do not match!")

# Combine training data
X_train = np.concatenate((train_infected_images, train_non_infected_images), axis=0)
y_train = np.concatenate((train_infected_labels, train_non_infected_labels), axis=0)

# Flatten and normalize training images
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten 2D images into 1D vectors
X_train_norm = X_train_flat.astype('float32') / 255.0  # Normalize pixel values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)

# Feature selection
selector = SelectKBest(score_func=f_classif, k="all")  
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Base models
base_models = [
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC(probability=True)),
    ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('rf', RandomForestClassifier()),
    ('adaboost', AdaBoostClassifier())
]

# Meta model
meta_model = RandomForestClassifier()

# Stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the model
stacking_model.fit(X_train_selected, y_train)

# ---------------- Separate Testing Phase ---------------- #

# Load test data (Separate test folder)
test_infected_images, test_infected_labels = load_images_from_folder(test_infected_path, 1)
test_non_infected_images, test_non_infected_labels = load_images_from_folder(test_non_infected_path, 0)

# Ensure test images have the same shape
if test_infected_images.shape[1:] != test_non_infected_images.shape[1:]:
    raise ValueError("Error: Test image dimensions do not match!")

# Combine test data
X_test = np.concatenate((test_infected_images, test_non_infected_images), axis=0)
y_test = np.concatenate((test_infected_labels, test_non_infected_labels), axis=0)

# Flatten and normalize test images
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_test_norm = X_test_flat.astype('float32') / 255.0

# Feature scaling and selection
X_test_scaled = scaler.transform(X_test_norm)
X_test_selected = selector.transform(X_test_scaled)

# Model evaluation on test data
y_pred = stacking_model.predict(X_test_selected)
y_pred_proba = stacking_model.predict_proba(X_test_selected)[:, 1]  # Probabilities for AUC

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)  # AUC Calculation

# Display results
print(f'**Evaluation on Separate Test Set:**')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC Score: {auc:.4f}')  # Print AUC score

