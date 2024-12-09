import cv2
import numpy as np
from sklearn.cluster import KMeans

# Step 1: Load and Preprocess the Images
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Segment the Images
def segment_image(image, segment_coordinates):
    height, width, _ = image.shape  # Get image dimensions
    x, y, w, h = segment_coordinates

    # Ensure valid width and height
    if w <= 0 or h <= 0:
        raise ValueError("Invalid segmentation dimensions: width and height must be greater than zero.")

    # Adjust coordinates to fit within image dimensions
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > width:
        w = width - x
    if y + h > height:
        h = height - y

    # Ensure a minimum region size to avoid zero or very small regions
    min_size = 10  # Minimum size for height and width
    if w < min_size:
        w = min_size if x + min_size <= width else width - x
    if h < min_size:
        h = min_size if y + min_size <= height else height - y

    return image[y:y + h, x:x + w]

# Step 3: Extract Dominant Colors using K-Means
def extract_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    
    if len(pixels) == 0:
        raise ValueError("Segmented image contains no pixels. Check the segment coordinates.")

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_

# Step 4: Compare to Predefined Color Ranges and Find Closest Match with Stages
def classify_color(test_color):
    r, g, b = test_color

    # Define expanded color ranges with stages for urine analysis
    color_ranges = {
        "Pale Yellow": {
            "stages": {
                "initial": {
                    "range": [(230, 255), (230, 255), (190, 230)],
                    "disease": "Well Hydrated",
                    "message": "Hydration is adequate.",
                    "suggestion": "Maintain hydration levels.",
                    "protein": "Negative",
                    "albumin": "Negative",
                    "creatinine": "Normal"
                },
                "progression": {
                    "range": [(210, 229), (210, 229), (170, 189)],
                    "disease": "Mild Dehydration",
                    "message": "Hydration is slightly low.",
                    "suggestion": "Increase fluid intake.",
                    "protein": "Negative",
                    "albumin": "Negative",
                    "creatinine": "Normal"
                },
                "severe": {
                    "range": [(200, 209), (200, 209), (150, 169)],
                    "disease": "Severe Dehydration",
                    "message": "Risk of severe dehydration.",
                    "suggestion": "Seek medical advice.",
                    "protein": "Trace",
                    "albumin": "Trace",
                    "creatinine": "Normal"
                }
            }
        },
        "Dark Yellow": {
            "stages": {
                "initial": {
                    "range": [(190, 199), (180, 199), (100, 149)],
                    "disease": "Dehydration",
                    "message": "Initial signs of dehydration.",
                    "suggestion": "Increase water intake.",
                    "protein": "Trace",
                    "albumin": "Negative",
                    "creatinine": "Normal"
                },
                "progression": {
                    "range": [(160, 189), (120, 179), (70, 99)],
                    "disease": "Moderate Dehydration",
                    "message": "Dehydration progressing.",
                    "suggestion": "Increase water immediately.",
                    "protein": "Positive",
                    "albumin": "Negative",
                    "creatinine": "Normal"
                },
                "severe": {
                    "range": [(150, 159), (100, 119), (50, 69)],
                    "disease": "Severe Dehydration",
                    "message": "High risk of dehydration-related complications.",
                    "suggestion": "Seek urgent medical help.",
                    "protein": "Positive",
                    "albumin": "Positive",
                    "creatinine": "Elevated"
                }
            }
        }
        # Additional color ranges can be added here...
    }

    # Iterate over each color and stage, and check the test color
    for color_name, stages in color_ranges.items():
        for stage, properties in stages['stages'].items():
            low_r, high_r = properties['range'][0]
            low_g, high_g = properties['range'][1]
            low_b, high_b = properties['range'][2]

            # Check if the color is within the defined range
            if low_r <= r <= high_r and low_g <= g <= high_g and low_b <= b <= high_b:
                return color_name, stage, properties['disease'], properties['message'], properties['suggestion'], {
                    "protein": properties["protein"],
                    "albumin": properties["albumin"],
                    "creatinine": properties["creatinine"]
                }

    # Fallback case: If no exact match, assume general condition with metrics
    return "General Condition", "No progression available", "Mild Condition Detected", "Increase hydration and seek medical advice if symptoms persist.", "Monitor health regularly", {
        "protein": "Normal",
        "albumin": "Normal",
        "creatinine": "Normal"
    }

# Main process
def urine_color_analysis(image_path, segment_coords):
    # Load and segment the image
    test_image = load_image(image_path)
    test_strip = segment_image(test_image, segment_coords)

    # Extract dominant color
    test_color = extract_dominant_color(test_strip, k=1)[0]
    test_color = [int(c) for c in test_color]  # Round color values to integers

    # Classify the color
    color_name, stage, disease, message, suggestion, metrics = classify_color(test_color)

    # Output the result in the terminal
    print(f"RGB Value: {test_color}")
    print(f"Associated Disease: {disease}")
    print(f"Suggestion: {suggestion}")
    print(f"Protein: {metrics['protein']}")
    print(f"Albumin: {metrics['albumin']}")
    print(f"Creatinine: {metrics['creatinine']}")
    print(f"Disease Progression: {stage} Stage")

# Example usage
if __name__ == "__main__":
    # Path to the image
    image_path = r"D:\Bm-hack\pr-2\eye2.jpg" # Update with the actual path to your image

    # Define the region for segmentation (x, y, width, height)
    segment_coords = (50, 200, 300, 100)  # Adjust these coordinates as needed

    # Perform the analysis and print the output
    urine_color_analysis(image_path, segment_coords)
