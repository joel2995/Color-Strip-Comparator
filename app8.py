import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Step 1: Load and Preprocess Images
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

# Function to match sclera color to associated disease and return detailed health metrics
def match_sclera_color(sclera_rgb, reference_colors):
    r, g, b = sclera_rgb

    # Define detailed health metrics for specific colors
    sclera_conditions = {
        "Light Red": {
            "disease": "Mild conjunctivitis, allergies",
            "suggestion": "Use over-the-counter antihistamines, avoid allergens.",
            "bilirubin": "Normal", "hemoglobin": "Normal", "protein": "Trace",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Normal", "dehydration": "Mild"
        },
        "Red": {
            "disease": "Moderate conjunctivitis, allergies",
            "suggestion": "Use anti-inflammatory drops, consult a doctor if symptoms persist.",
            "bilirubin": "Normal", "hemoglobin": "Slightly Low", "protein": "Elevated",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Elevated", "dehydration": "Moderate"
        },
        "Yellow (Early Jaundice)": {
            "disease": "Early jaundice, mild liver disease",
            "suggestion": "Hydrate well and consult a physician for liver function tests.",
            "bilirubin": "1.2 mg/dL", "hemoglobin": "Normal", "protein": "Normal",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Normal", "dehydration": "Mild"
        },
        "Yellow (Severe Liver Disease)": {
            "disease": "Severe liver disease",
            "suggestion": "Seek immediate medical attention.",
            "bilirubin": "5.0 mg/dL", "hemoglobin": "Normal", "protein": "Normal",
            "blood_sugar": "Normal", "cholesterol": "High", "crp": "Normal", "dehydration": "Severe"
        }
    }

    # Match detected sclera color with reference colors using closest match by Euclidean distance
    closest_color_name = None
    closest_distance = float('inf')

    for index, reference_color in enumerate(reference_colors):
        color_name = list(sclera_conditions.keys())[index]
        dist = distance.euclidean(sclera_rgb, reference_color)

        if dist < closest_distance:
            closest_distance = dist
            closest_color_name = color_name

    condition = sclera_conditions.get(closest_color_name, {
        "disease": "General Eye Irritation",
        "suggestion": "Consult a doctor for further evaluation.",
        "bilirubin": "Normal", "hemoglobin": "Normal", "protein": "Normal",
        "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Normal", "dehydration": "Mild"
    })

    return (
        condition["disease"],
        condition["suggestion"],
        condition["bilirubin"],
        condition["hemoglobin"],
        condition["protein"],
        condition["blood_sugar"],
        condition["cholesterol"],
        condition["crp"],
        condition["dehydration"],
        closest_color_name  # Matched color name
    )

# Function to check disease progression based on RGB value
def check_eye_color_disease(sclera_rgb):
    r, g, b = sclera_rgb

    # Define the color ranges and corresponding multiple diseases
    diseases = [
        {
            "color": "Yellow",
            "initial": (200, 255, 150),
            "progression": (255, 255, 100),
            "severe": (204, 204, 0),
            "initial_diseases": ["Jaundice", "Liver Disease", "Gallbladder Issues"],
            "progression_diseases": ["Severe Jaundice", "Liver Failure", "Pancreatitis"],
            "severe_diseases": ["Advanced Liver Disease", "Hepatitis", "Cirrhosis"]
        },
        {
            "color": "Red",
            "initial": (200, 255, 150),
            "progression": (255, 153, 153),
            "severe": (255, 102, 102),
            "initial_diseases": ["Conjunctivitis", "Minor Irritation"],
            "progression_diseases": ["Uveitis", "Severe Allergies"],
            "severe_diseases": ["Hemorrhage", "Glaucoma", "Severe Inflammation"]
        }
    ]
    
    # Determine the disease progression stage
    for disease in diseases:
        if disease["initial"] <= sclera_rgb <= disease["progression"]:
            return f"Initial Stage Diseases: {', '.join(disease['initial_diseases'])}"
        elif disease["progression"] <= sclera_rgb <= disease["severe"]:
            return f"Progression Stage Diseases: {', '.join(disease['progression_diseases'])}"
        elif sclera_rgb >= disease["severe"]:
            return f"Severe Stage Diseases: {', '.join(disease['severe_diseases'])}"
    
    return "General eye irritation detected. Consult an ophthalmologist."

# Function to detect sclera and match diseases
def detect_eye_sclera(image_path, reference_colors):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")
    
    # Convert to grayscale for eye detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) == 0:
        return "No eyes detected."

    sclera_found = False
    sclera_rgb = None

    for (x, y, w, h) in eyes:
        eye_region = image[y:y+h, x:x+w]

        # Convert to HSV for better color segmentation
        hsv_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

        # Define the range for sclera color
        lower_sclera = np.array([0, 50, 200])
        upper_sclera = np.array([180, 255, 255])

        # Create a mask for the sclera
        mask = cv2.inRange(hsv_eye, lower_sclera, upper_sclera)

        # Calculate the average color of the sclera area
        sclera_area = cv2.bitwise_and(eye_region, eye_region, mask=mask)
        mean_color = cv2.mean(sclera_area, mask=mask)[:3]  # Get BGR values

        # Convert to RGB
        sclera_rgb = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))

        if np.count_nonzero(mask) > 0:
            sclera_found = True

    if sclera_found:
        # Match the sclera color to a disease and get associated health metrics
        detailed_result = match_sclera_color(sclera_rgb, reference_colors)
        disease_progression = check_eye_color_disease(sclera_rgb)
        return sclera_rgb, *detailed_result, disease_progression
    else:
        return "No sclera color detected."

# Example usage
if __name__ == "__main__":
    # Define the path to the parent strip (color reference) image
    parent_image_path = r"D:\Bm-hack\pr-2\Parentst.jpg"  # Update this path to the parent strip image
    
    # Load the parent image and extract reference colors
    reference_colors = extract_reference_colors(parent_image_path, num_colors=4)
    print("Extracted Reference Colors (RGB):", reference_colors)

    # Define the path to the eye image to process
    image_path = r"D:\Bm-hack\pr-2\480px-Urine_sample.jpeg" # Update this to your eye image

    # Perform the analysis
    result = detect_eye_sclera(image_path, reference_colors)

    # Display the result
    if isinstance(result, tuple):
        rgb_value, disease_message, suggestion, bilirubin, hemoglobin, protein, blood_sugar, cholesterol, crp, dehydration, matched_color, progression = result
        print(f"RGB Value: {rgb_value}")
        print(f"Color Matched: {matched_color}")
        print(f"Associated Disease: {disease_message}")
        print(f"Suggestion: {suggestion}")
        print(f"Bilirubin: {bilirubin}")
        print(f"Hemoglobin: {hemoglobin}")
        print(f"Protein: {protein}")
        print(f"Blood Sugar: {blood_sugar}")
        print(f"Cholesterol: {cholesterol}")
        print(f"CRP (Inflammation Marker): {crp}")
        print(f"Dehydration Level: {dehydration}")
        print(f"Disease Progression: {progression}")
    else:
        print(result)
