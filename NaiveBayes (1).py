import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# Function to load images from two folders (infected and not infected)
def load_binary_classification_dataset(infected_folder, not_infected_folder):
    images = []
    labels = []

    # Load infected images (label 1)
    for filename in os.listdir(infected_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(infected_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(1)  # Infected

    # Load not infected images (label 0)
    for filename in os.listdir(not_infected_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(not_infected_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(0)  # Not infected

    # Find the minimum shape among all images
    min_shape = min((img.shape for img in images))

    # Resize all images to the minimum shape
    images_resized = [cv2.resize(img, (min_shape[1], min_shape[0])) for img in images]

    return np.array(images_resized), np.array(labels)

# Paths for training and testing datasets
train_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/train/infected"
train_not_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/train/notinfected"
test_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/test/infected"
test_not_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/test/notinfected"

# Load training and testing images
X_train, y_train = load_binary_classification_dataset(train_infected_path, train_not_infected_path)
X_test, y_test = load_binary_classification_dataset(test_infected_path, test_not_infected_path)

# Flatten and normalize the images
X_train_flat = [img.flatten() for img in X_train]
X_test_flat = [img.flatten() for img in X_test]

X_train_normalized = np.array(X_train_flat).astype('float32') / 255.0
X_test_normalized = np.array(X_test_flat).astype('float32') / 255.0

# Feature selection using K=ALL (all features are selected)
X_train_selected = X_train_normalized  # No feature selection
X_test_selected = X_test_normalized

# Train Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_selected, y_train)

# Evaluate model performance
y_pred = nb_model.predict(X_test_selected)
final_accuracy = nb_model.score(X_test_selected, y_test)

# Display accuracy
print(f"Final Accuracy on Test Set (K=ALL): {final_accuracy}")

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Calculate AUC-ROC
y_pred_proba = nb_model.predict_proba(X_test_selected)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC on Test Set (K=ALL): {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naïve Bayes (K=ALL)')
plt.legend(loc='lower right')
plt.show()
