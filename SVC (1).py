import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Function to load dataset
def load_binary_classification_dataset(infected_folder, not_infected_folder):
    images, labels = [], []

    for filename in os.listdir(infected_folder):
        img_path = os.path.join(infected_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(1)  # Infected

    for filename in os.listdir(not_infected_folder):
        img_path = os.path.join(not_infected_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(0)  # Not infected

    min_shape = min((img.shape for img in images))
    images_resized = [cv2.resize(img, (min_shape[1], min_shape[0])) for img in images]

    return np.array(images_resized), np.array(labels)

# Load dataset
X_train, y_train = load_binary_classification_dataset("/content/drive/MyDrive/SUDESH_PERSONAL/train/infected", "/content/drive/MyDrive/SUDESH_PERSONAL/train/notinfected")
X_test, y_test = load_binary_classification_dataset("/content/drive/MyDrive/SUDESH_PERSONAL/test/infected", "/content/drive/MyDrive/SUDESH_PERSONAL/test/notinfected")

# Flatten and normalize
X_train_flat = [img.flatten() for img in X_train]
X_test_flat = [img.flatten() for img in X_test]

X_train_norm = np.array(X_train_flat).astype('float32') / 255.0
X_test_norm = np.array(X_test_flat).astype('float32') / 255.0

# Feature selection using ANOVA F-test (K="ALL" -> select all features)
X_train_selected = X_train_norm
X_test_selected = X_test_norm

# Train SVC with best kernel using cross-validation
best_kernel = None
best_score = 0
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    svc = SVC(kernel=kernel, probability=True)
    scores = cross_val_score(svc, X_train_selected, y_train, cv=5)
    avg_score = scores.mean()
    
    if avg_score > best_score:
        best_score = avg_score
        best_kernel = kernel

print(f"Best Kernel: {best_kernel}")

# Train final model
svc_model = SVC(kernel=best_kernel, probability=True)
svc_model.fit(X_train_selected, y_train)

# Evaluate
y_pred = svc_model.predict(X_test_selected)
y_pred_proba = svc_model.predict_proba(X_test_selected)[:, 1]

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC-ROC: {roc_auc:.2f}")

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend()
plt.show()
