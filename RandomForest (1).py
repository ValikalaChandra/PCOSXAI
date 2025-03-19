import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
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
X_train, y_train = load_binary_classification_dataset("/kaggle/input/pcos-detection-using-ultrasound-images/data/train/infected", "/kaggle/input/pcos-detection-using-ultrasound-images/data/train/notinfected")
X_test, y_test = load_binary_classification_dataset("/kaggle/input/pcos-detection-using-ultrasound-images/data/test/infected", "/kaggle/input/pcos-detection-using-ultrasound-images/data/test/notinfected")

# Flatten and normalize
X_train_flat = [img.flatten() for img in X_train]
X_test_flat = [img.flatten() for img in X_test]

X_train_norm = np.array(X_train_flat).astype('float32') / 255.0
X_test_norm = np.array(X_test_flat).astype('float32') / 255.0

# Feature selection using ANOVA F-test (K="ALL" -> select all features)
X_train_selected = X_train_norm
X_test_selected = X_test_norm
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest with cross-validation
best_n_estimators = None
best_score = 0

for n in [50, 100, 150]:
    rf = RandomForestClassifier(n_estimators=n)
    scores = cross_val_score(rf, X_train_selected, y_train, cv=5)
    avg_score = scores.mean()
    
    if avg_score > best_score:
        best_score = avg_score
        best_n_estimators = n

print(f"Best n_estimators: {best_n_estimators}")

# Train final Random Forest model
rf_model = RandomForestClassifier(n_estimators=best_n_estimators)
rf_model.fit(X_train_selected, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_selected)
y_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {roc_auc:.2f}")
# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.legend()
plt.show()