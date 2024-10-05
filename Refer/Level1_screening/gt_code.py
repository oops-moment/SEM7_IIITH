
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# Path setup (Replace with actual paths)
court_img_path = 'output.jpg'
top_players_path = 'two_players_top/'
bottom_players_path = 'two_players_bot/'
output_folder = 'output_players/'

# Create output directory if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create subfolders for each player
for i in range(1, 5):
    os.makedirs(f'{output_folder}/player_{i}', exist_ok=True)

# Function to extract average background color (assumed to be the most common in the image)
def extract_average_color(image):
    h, w, _ = image.shape
    image_reshaped = image.reshape((h * w, 3))
    avg_color = np.mean(image_reshaped, axis=0)
    return avg_color

# Function to subtract the background (average court color)
def subtract_average_color(image, avg_color):
    # Convert the image to float32 for more accurate subtraction
    image_float = np.float32(image)
    avg_color_float = np.float32(avg_color)

    # Subtract the average background color from the image
    subtracted = cv2.subtract(image_float, avg_color_float)

    # Clip values to stay within 0-255 and convert back to uint8
    subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)
    
    return subtracted

# Function to extract dominant colors from an image using KMeans clustering
def extract_dominant_colors(image, k=2):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans clustering to find k dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Return the cluster colors
    return colors.astype(int)

# Function to classify images into player folders based on dominant color separation
def classify_and_save(image_path, color_groups):
    img = cv2.imread(image_path)
    
    # Extract the average background color from the image
    avg_color = extract_average_color(img)
    
    # Subtract the background color
    fg_img = subtract_average_color(img, avg_color)
    
    # Extract the dominant colors of the foreground image (player)
    extracted_colors = extract_dominant_colors(fg_img, k=2)
    
    # Compare the dominant colors with current color groups and classify accordingly
    for i, group in enumerate(color_groups):
        for extracted_color in extracted_colors:
            if np.linalg.norm(extracted_color - group) < 50:  # Threshold for color similarity
                return i + 1
    
    return None

# Function to process images in a folder and segregate them based on color
def process_images(player_folder, color_groups):
    for filename in os.listdir(player_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(player_folder, filename)
            
            # Classify image based on color comparison
            player_class = classify_and_save(img_path, color_groups)
            
            if player_class:
                output_path = os.path.join(output_folder, f'player_{player_class}', filename)
                img = cv2.imread(img_path)
                cv2.imwrite(output_path, img)
            else:
                print(f"Could not classify image: {filename}")

# Generate the initial color groups (reference colors for segregation)
def initialize_color_groups():
    color_groups = []
    # Process some initial images to extract dominant colors (for each player)
    example_images = [os.path.join(top_players_path, f) for f in os.listdir(top_players_path)[:2]] + \
                     [os.path.join(bottom_players_path, f) for f in os.listdir(bottom_players_path)[:2]]
    
    for img_path in example_images:
        img = cv2.imread(img_path)
        dominant_colors = extract_dominant_colors(img, k=2)
        color_groups.append(dominant_colors[0])  # Assume each player has a distinct color for their shirt

    return color_groups

# Main driver function
def main():
    # Initialize color groups from example images
    color_groups = initialize_color_groups()
    
    # Process top and bottom player folders
    print("Processing top two players...")
    process_images(top_players_path, color_groups[:2])
    
    print("Processing bottom two players...")
    process_images(bottom_players_path, color_groups[2:])

if __name__ == "__main__":
    main()