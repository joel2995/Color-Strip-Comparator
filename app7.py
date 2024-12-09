import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Step 1: Load and Preprocess the Images
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Extract Reference Colors from Parent Strip
def extract_reference_colors(parent_image_path, num_colors=6):
    parent_image = load_image(parent_image_path)
    dominant_colors = extract_dominant_color(parent_image, k=num_colors)
    return [tuple(map(int, color)) for color in dominant_colors]  # Convert to integers

# Extract dominant colors from an image using KMeans
def extract_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    
    if len(pixels) == 0:
        raise ValueError("Segmented image contains no pixels.")

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_

# Function to match urine color to associated disease and return detailed health metrics
def match_urine_color(urine_rgb, reference_colors):
    r, g, b = urine_rgb

    # Define detailed health metrics for specific colors
    urine_conditions = {
        "Pale Yellow": {
            "disease": "Well Hydrated",
            "suggestion": "Maintain hydration levels.",
            "protein": "Negative",
            "albumin": "Negative",
            "creatinine": "Normal"
        },
        "Dark Yellow": {
            "disease": "Dehydration",
            "suggestion": "Increase water intake.",
            "protein": "Trace",
            "albumin": "Negative",
            "creatinine": "Normal"
        },
        "Amber": {
            "disease": "Moderate Dehydration",
            "suggestion": "Increase water intake immediately.",
            "protein": "Positive",
            "albumin": "Negative",
            "creatinine": "Normal"
        },
        "Brown": {
            "disease": "Severe Dehydration or possible liver issues",
            "suggestion": "Seek medical advice immediately.",
            "protein": "Positive",
            "albumin": "Positive",
            "creatinine": "Elevated"
        }
    }

    # Match detected urine color with reference colors using closest match by Euclidean distance
    closest_color_name = None
    closest_distance = float('inf')

    for index, reference_color in enumerate(reference_colors):
        color_name = list(urine_conditions.keys())[index]
        dist = distance.euclidean(urine_rgb, reference_color)

        if dist < closest_distance:
            closest_distance = dist
            closest_color_name = color_name

    condition = urine_conditions.get(closest_color_name, {
        "disease": "General Condition",
        "suggestion": "Increase hydration and seek medical advice if symptoms persist.",
        "protein": "Normal",
        "albumin": "Normal",
        "creatinine": "Normal"
    })

    return (
        condition["disease"],
        condition["suggestion"],
        condition["protein"],
        condition["albumin"],
        condition["creatinine"],
        closest_color_name  # Matched color name
    )

# Function to determine the disease progression based on the urine color
def check_urine_color_disease(urine_rgb):
    r, g, b = urine_rgb

    # Define color ranges and corresponding multiple diseases
    diseases = [
        {
            "color": "Pale Yellow",
            "initial": (230, 255, 190),
            "progression": (210, 229, 170),
            "severe": (200, 209, 150),
            "initial_diseases": ["Well Hydrated"],
            "progression_diseases": ["Mild Dehydration"],
            "severe_diseases": ["Severe Dehydration"]
        },
        {
            "color": "Dark Yellow",
            "initial": (190, 199, 100),
            "progression": (160, 189, 70),
            "severe": (150, 159, 50),
            "initial_diseases": ["Dehydration"],
            "progression_diseases": ["Moderate Dehydration"],
            "severe_diseases": ["Severe Dehydration or Liver Issue"]
        }
    ]

    # Determine the disease progression stage
    for disease in diseases:
        if (r >= disease["initial"][0] and g >= disease["initial"][1] and b >= disease["initial"][2] and
            r <= disease["progression"][0] and g <= disease["progression"][1] and b <= disease["progression"][2]):
            return f"Initial Stage Diseases: {', '.join(disease['initial_diseases'])}"
        elif (r >= disease["progression"][0] and g >= disease["progression"][1] and b >= disease["progression"][2] and
              r <= disease["severe"][0] and g <= disease["severe"][1] and b <= disease["severe"][2]):
            return f"Progression Stage Diseases: {', '.join(disease['progression_diseases'])}"
        elif (r >= disease["severe"][0] and g >= disease["severe"][1] and b >= disease["severe"][2]):
            return f"Severe Stage Diseases: {', '.join(disease['severe_diseases'])}"

    return "General condition detected. Increase hydration."

# Function to detect urine strip color and match diseases
def urine_color_analysis(image_path, reference_colors, segment_coords):
    # Load and segment the image
    test_image = load_image(image_path)
    test_strip = segment_image(test_image, segment_coords)

    # Extract dominant color
    test_color = extract_dominant_color(test_strip, k=1)[0]
    test_color = [int(c) for c in test_color]  # Round color values to integers

    # Match the urine color to a disease and get associated health metrics
    detailed_result = match_urine_color(test_color, reference_colors)
    disease_progression = check_urine_color_disease(test_color)
    return test_color, *detailed_result, disease_progression

# Segment the image based on coordinates
def segment_image(image, segment_coordinates):
    height, width, _ = image.shape
    x, y, w, h = segment_coordinates

    if w <= 0 or h <= 0:
        raise ValueError("Invalid segmentation dimensions.")

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > width:
        w = width - x
    if y + h > height:
        h = height - y

    return image[y:y + h, x:x + w]

# Example usage
if __name__ == "__main__":
    # Define the path to the parent strip (color reference) image
    parent_image_path = r"D:\Bm-hack\pr-2\Parentst.jpg"  # Update this path to the parent strip image

    # Load the parent image and extract reference colors
    reference_colors = extract_reference_colors(parent_image_path, num_colors=4)
    print("Extracted Reference Colors (RGB):", reference_colors)

    # Define the path to the urine image to process
    image_path = r"D:\Bm-hack\pr-2\480px-Urine_sample.jpeg"  # Update this to your urine image

    # Define the region for segmentation (x, y, width, height)
    segment_coords = (50, 200, 300, 100)  # Adjust these coordinates as needed

    # Perform the analysis
    result = urine_color_analysis(image_path, reference_colors, segment_coords)

    # Display the result
    if isinstance(result, tuple):
        rgb_value, disease_message, suggestion, protein, albumin, creatinine, matched_color, progression = result
        print(f"RGB Value: {rgb_value}")
        print(f"Color Matched: {matched_color}")
        print(f"Associated Disease: {disease_message}")
        print(f"Suggestion: {suggestion}")
        print(f"Protein: {protein}")
        print(f"Albumin: {albumin}")
        print(f"Creatinine: {creatinine}")
        print(f"Disease Progression: {progression}")
    else:
        print(result)
