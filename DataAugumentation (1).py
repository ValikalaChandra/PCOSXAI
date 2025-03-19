import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Define dataset paths
base_input_path = "/kaggle/input/pcos-detection-using-ultrasound-images/data/train"
infected_folder = os.path.join(base_input_path, "infected")
non_infected_folder = os.path.join(base_input_path, "notinfected")

# Define output paths for augmented images
base_output_path = "/kaggle/working"
augmented_infected_folder = os.path.join(base_output_path, "Augmented_Infected")
augmented_non_infected_folder = os.path.join(base_output_path, "Augmented_NonInfected")

# Create output directories if they don't exist
os.makedirs(augmented_infected_folder, exist_ok=True)
os.makedirs(augmented_non_infected_folder, exist_ok=True)

# Count original images
num_infected = len([f for f in os.listdir(infected_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
num_non_infected = len([f for f in os.listdir(non_infected_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

# Target balanced dataset size
target_size = 12567

# Compute required augmentations
num_augmented_infected = max(0, target_size - num_infected)
num_augmented_non_infected = max(0, target_size - num_non_infected)

# Compute augmentations per image
infected_aug_per_image = max(1, num_augmented_infected // num_infected)
non_infected_aug_per_image = max(1, num_augmented_non_infected // num_non_infected)

# GPU-Powered Data Augmentation using Torchvision
augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=20, shear=15),
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

def augment_and_save_images(input_folder, output_folder, num_augmented):
    """Augments images from input_folder and saves them to output_folder using GPU."""
    
    for filename in tqdm(os.listdir(input_folder), desc=f"Processing {input_folder}"):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(input_folder, filename)

            # Load image using PIL
            img = Image.open(img_path).convert("RGB")

            # Move image to GPU
            img_tensor = transforms.ToTensor()(img).to(device)

            # Save original image
            original_save_path = os.path.join(output_folder, filename)
            vutils.save_image(img_tensor, original_save_path)

            # Generate augmented images and save them
            for i in range(num_augmented):
                aug_img_tensor = augmentation_pipeline(img).to(device)
                aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                aug_save_path = os.path.join(output_folder, aug_filename)
                vutils.save_image(aug_img_tensor, aug_save_path)

    print(f" Augmentation completed for {input_folder}. Images saved in {output_folder}")

# Apply augmentation to infected and non-infected images
augment_and_save_images(infected_folder, augmented_infected_folder, infected_aug_per_image)
augment_and_save_images(non_infected_folder, augmented_non_infected_folder, non_infected_aug_per_image)

# Print paths for reference
print("\n Dataset Prepared:")
print(" Infected Images Path:", infected_folder)
print(" Non-Infected Images Path:", non_infected_folder)
print(" Augmented Infected Images Path:", augmented_infected_folder)
print(" Augmented Non-Infected Images Path:", augmented_non_infected_folder)
