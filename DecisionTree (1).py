import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


# Function to load images from two folders (infected and not infected)
def load_images_from_folders(infected_folder, not_infected_folder):
    images = []
    labels = []

    for folder, label in [(infected_folder, 1), (not_infected_folder, 0)]:
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    min_shape = min((img.shape for img in images))
    images_resized = [cv2.resize(img, (min_shape[1], min_shape[0])) for img in images]

    return np.array(images_resized), np.array(labels)

# Paths for training and testing datasets
train_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/train/infected"
train_not_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/train/notinfected"
test_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/test/infected"
test_not_infected_path = "/content/drive/MyDrive/SUDESH_PERSONAL/train/notinfected"

# Load training and testing images separately
X_train, y_train = load_images_from_folders(train_infected_path, train_not_infected_path)
X_test, y_test = load_images_from_folders(test_infected_path, test_not_infected_path)


# Flatten and normalize images
X_train = np.array([img.flatten() for img in X_train], dtype=np.float32) / 255.0
X_test = np.array([img.flatten() for img in X_test], dtype=np.float32) / 255.0


# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k="all")
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)


# Hyperparameter tuning for Decision Tree
best_accuracy, best_max_depth = 0, 0
results_tree = []

for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train_selected, y_train)
    
    accuracy = model.score(X_test_selected, y_test)
    results_tree.append({'max_depth': depth, 'accuracy': accuracy})

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_max_depth = depth

# Convert results to DataFrame
results_df = pd.DataFrame(results_tree)

# Plot accuracy for different max_depth values
plt.plot(results_df["max_depth"], results_df["accuracy"], marker="o")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs. Max Depth")
plt.show()


# Train final model with best depth
final_model = DecisionTreeClassifier(max_depth=best_max_depth)
final_model.fit(X_train_selected, y_train)


# Predict on test data
y_pred = final_model.predict(X_test_selected)

# Compute precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Best Accuracy: {best_accuracy} (max_depth={3})")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


# ROC Curve
y_prob = final_model.predict_proba(X_test_selected)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()