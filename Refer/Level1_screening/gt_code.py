import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

# Paths to input folders
top_folder = 'two_players_top'
bot_folder = 'two_players_bot'
court_image_path = 'output.jpg'

# Create output folder structure
output_dir = 'output'
players = ['player1', 'player2', 'player3', 'player4']
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Clean existing output folder
os.makedirs(output_dir)
for player in players:
    os.makedirs(os.path.join(output_dir, player))

# Function to perform background subtraction using absolute difference
def subtract_background(img, background):
    # Resize background to match image size if necessary
    background = cv2.resize(background, (img.shape[1], img.shape[0]))
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Perform absolute difference
    diff = cv2.absdiff(img_gray, background_gray)
    
    # Threshold the difference to obtain a binary mask (players vs background)
    _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Optional: Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Function to process images, subtract background, and classify players
def process_images(folder_path, player1_folder, player2_folder, court_image):
    images = os.listdir(folder_path)
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        # Step 1: Background subtraction
        mask = subtract_background(img, court_image)
        
        # Step 2: Apply the mask to get non-background (player) pixels
        player_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Step 3: Reshape the non-zero pixels for clustering (no modification to the original image)
        non_zero_pixels = player_img[mask > 0].reshape((-1, 3))  # Only player pixels
        
        # Step 4: Apply KMeans to classify players (2 clusters for 2 players)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(non_zero_pixels)
        labels = kmeans.labels_
        
        # Step 5: Assign the image to player 1 or player 2 based on the dominant cluster
        # Count the labels to determine which player is dominant in the image
        label_counts = np.bincount(labels)
        dominant_label = np.argmax(label_counts)  # The dominant player in the image
        
        # Classify the image into the correct folder without modifying it
        if dominant_label == 0:
            shutil.copy(img_path, os.path.join(player1_folder, img_name))
        else:
            shutil.copy(img_path, os.path.join(player2_folder, img_name))

# Load court background image
court_image = cv2.imread(court_image_path)
if court_image is None:
    print(f"Error: Could not load court image from {court_image_path}")
    exit(1)

# Process the top and bottom folder images without modifying them
process_images(top_folder, os.path.join(output_dir, 'player1'), os.path.join(output_dir, 'player2'), court_image)
process_images(bot_folder, os.path.join(output_dir, 'player3'), os.path.join(output_dir, 'player4'), court_image)

print("Classification complete! Original images are moved into their respective folders.")