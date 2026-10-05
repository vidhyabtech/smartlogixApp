import os
import cv2

# Define paths based on your current workspace structure
base_dir = "./drone_images"
categories = ["battery_damage", "motor_damage", "total damage"]

def augment_images_in_folder(folder_name, target_count=100):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    images = os.listdir(folder_path)
    images = [img for img in images if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    current_count = len(images)
    print(f"Processing '{folder_name}': Found {current_count} original images.")

    if current_count == 0:
        return

    i = 0
    while current_count < target_count:
        for img_name in images:
            if current_count >= target_count:
                break
            
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Apply simple variations: flips and rotations
            if i % 3 == 0:
                augmented = cv2.flip(img, 1)  # Horizontal flip
            elif i % 3 == 1:
                augmented = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                augmented = cv2.flip(img, 0)  # Vertical flip

            new_filename = f"aug_{current_count}_{img_name}"
            cv2.imwrite(os.path.join(folder_path, new_filename), augmented)
            current_count += 1
            i += 1

    print(f"Finished! '{folder_name}' now has {current_count} images.\n")

# Run augmentation for your sparse categories
for category in categories:
    augment_images_in_folder(category, target_count=100)