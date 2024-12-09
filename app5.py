import cv2
import numpy as np

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to match sclera color to associated disease and return detailed health metrics
def match_sclera_color(sclera_rgb):
    r, g, b = sclera_rgb
    
    # Define the ranges, diseases, health metrics, and suggestions
    sclera_conditions = [
        {
            "color": "Light Red",
            "r_range": (200, 255), "g_range": (150, 255), "b_range": (150, 255),
            "disease": "Mild conjunctivitis, allergies",
            "suggestion": "Use over-the-counter antihistamines, avoid allergens.",
            "bilirubin": "Normal", "hemoglobin": "Normal", "protein": "Trace",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Normal", "dehydration": "Mild"
        },
        {
            "color": "Red",
            "r_range": (200, 255), "g_range": (100, 200), "b_range": (100, 200),
            "disease": "Moderate conjunctivitis, allergies",
            "suggestion": "Use anti-inflammatory drops, consult a doctor if symptoms persist.",
            "bilirubin": "Normal", "hemoglobin": "Slightly Low", "protein": "Elevated",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Elevated", "dehydration": "Moderate"
        },
        {
            "color": "Yellow",
            "r_range": (200, 255), "g_range": (200, 255), "b_range": (100, 255),
            "disease": "Early jaundice, mild liver disease",
            "suggestion": "Hydrate well and consult a physician for liver function tests.",
            "bilirubin": "1.2 mg/dL", "hemoglobin": "Normal", "protein": "Normal",
            "blood_sugar": "Normal", "cholesterol": "Normal", "crp": "Normal", "dehydration": "Mild"
        },
        {
            "color": "Yellow",
            "r_range": (200, 255), "g_range": (200, 255), "b_range": (0, 100),
            "disease": "Severe liver disease",
            "suggestion": "Seek immediate medical attention.",
            "bilirubin": "5.0 mg/dL", "hemoglobin": "Normal", "protein": "Normal",
            "blood_sugar": "Normal", "cholesterol": "High", "crp": "Normal", "dehydration": "Severe"
        }
        # Add more conditions or expand ranges as needed...
    ]
    
    # Ensure every RGB value maps to a known condition
    for condition in sclera_conditions:
        if condition["r_range"][0] <= r <= condition["r_range"][1] and \
           condition["g_range"][0] <= g <= condition["g_range"][1] and \
           condition["b_range"][0] <= b <= condition["b_range"][1]:
            return (
                condition["disease"],
                condition["suggestion"],
                condition["bilirubin"],
                condition["hemoglobin"],
                condition["protein"],
                condition["blood_sugar"],
                condition["cholesterol"],
                condition["crp"],
                condition["dehydration"]
            )
    
    # If no specific match is found, return a default health condition
    return (
        "General Eye Irritation",
        "Consult a doctor for further evaluation.",
        "Normal", "Normal", "Normal", "Normal", "Normal", "Normal", "Mild"
    )

# Function to check disease progression based on RGB value
def check_eye_color_disease(rgb):
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
    
    for disease in diseases:
        if disease["initial"] <= rgb <= disease["progression"]:
            return f"Initial Stage Diseases: {', '.join(disease['initial_diseases'])}"
        elif disease["progression"] <= rgb <= disease["severe"]:
            return f"Progression Stage Diseases: {', '.join(disease['progression_diseases'])}"
        elif rgb >= disease["severe"]:
            return f"Severe Stage Diseases: {', '.join(disease['severe_diseases'])}"
    
    return "General eye irritation detected. Consult an ophthalmologist."

# Function to detect sclera and match diseases
def detect_eye_sclera(image_path):
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

    sclera_found = False  # Flag to check if sclera was detected
    sclera_rgb = None

    for (x, y, w, h) in eyes:
        # Define the region of interest around the eye (sclera)
        eye_region = image[y:y+h, x:x+w]

        # Convert to HSV for better color segmentation
        hsv_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

        # Define the range for sclera color (may need adjustment based on eye condition)
        lower_sclera = np.array([0, 50, 200])  # Adjusted lower threshold for light colors in HSV
        upper_sclera = np.array([180, 255, 255])  # Adjusted upper threshold for light colors in HSV

        # Create a mask for the sclera
        mask = cv2.inRange(hsv_eye, lower_sclera, upper_sclera)

        # Calculate the average color of the sclera area
        sclera_area = cv2.bitwise_and(eye_region, eye_region, mask=mask)
        
        # Get the mean color values in BGR format
        mean_color = cv2.mean(sclera_area, mask=mask)[:3]  # Get BGR values

        # Convert to RGB
        sclera_rgb = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))  # Convert to RGB

        # Check if the sclera area is predominantly white or light
        if np.count_nonzero(mask) > 0:
            sclera_found = True

    if sclera_found:
        # Match the sclera color to a disease and get associated health metrics
        detailed_result = match_sclera_color(sclera_rgb)
        progression_result = check_eye_color_disease(sclera_rgb)
        return sclera_rgb, *detailed_result, progression_result
    else:
        return "No sclera color detected."

# Example usage
image_path = r"D:\Bm-hack\pr-2\eye1.jpg"  # Update with the actual image path
result = detect_eye_sclera(image_path)

if isinstance(result, tuple):
    rgb_value, disease_message, suggestion, bilirubin, hemoglobin, protein, blood_sugar, cholesterol, crp, dehydration, disease_progression = result
    print(f"RGB Value: {rgb_value}")
    print(f"Associated Disease: {disease_message}")
    print(f"Suggestion: {suggestion}")
    print(f"Bilirubin: {bilirubin}")
    print(f"Hemoglobin: {hemoglobin}")
    print(f"Protein: {protein}")
    print(f"Blood Sugar: {blood_sugar}")
    print(f"Cholesterol: {cholesterol}")
    print(f"CRP (Inflammation Marker): {crp}")
    print(f"Dehydration Level: {dehydration}")
    print(f"Disease Progression: {disease_progression}")
else:
    print(result)
