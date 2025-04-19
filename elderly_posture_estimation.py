import cv2
import numpy as np
import os
import math
import sys
import traceback

# Try to import MediaPipe, but handle the case where it's not installed
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clamp the value to avoid numerical errors
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

def analyze_single_image(image_path, output_directory):
    """Process a single image with accurate pose detection and measurements."""
    try:
        # Ensure output directory exists
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist at {image_path}")
            return False
        
        # Read the image
        print(f"Reading image from {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return False
        
        print(f"Image loaded successfully. Dimensions: {image.shape}")
        
        # Create a copy for drawing
        result_image = image.copy()
        
        # Use MediaPipe if available, otherwise use the fallback method
        if MEDIAPIPE_AVAILABLE:
            success = process_with_mediapipe(image, result_image, image_path, output_directory)
            if not success:
                print("MediaPipe detection failed. Using fallback method.")
                return process_with_fallback(image, result_image, image_path, output_directory)
            return success
        else:
            print("MediaPipe not available. Using fallback method.")
            return process_with_fallback(image, result_image, image_path, output_directory)
            
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return False

def process_with_mediapipe(image, result_image, image_path, output_directory):
    """Process image using MediaPipe pose detection."""
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        # Setup pose detection
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
            
            # Convert to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            pose_results = pose.process(image_rgb)
            
            # Check if pose detection was successful
            if not pose_results.pose_landmarks:
                return False
                
            # Extract landmarks
            landmarks = {}
            landmark_points = pose_results.pose_landmarks.landmark
            
            # Get image dimensions
            height, width, _ = image.shape
            
            # Map relevant landmarks
            landmarks["nose"] = (int(landmark_points[mp_pose.PoseLandmark.NOSE].x * width),
                               int(landmark_points[mp_pose.PoseLandmark.NOSE].y * height))
            
            landmarks["left_ear"] = (int(landmark_points[mp_pose.PoseLandmark.LEFT_EAR].x * width),
                                   int(landmark_points[mp_pose.PoseLandmark.LEFT_EAR].y * height))
            
            landmarks["right_ear"] = (int(landmark_points[mp_pose.PoseLandmark.RIGHT_EAR].x * width),
                                    int(landmark_points[mp_pose.PoseLandmark.RIGHT_EAR].y * height))
            
            landmarks["left_shoulder"] = (int(landmark_points[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                                        int(landmark_points[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
            
            landmarks["right_shoulder"] = (int(landmark_points[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                                         int(landmark_points[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
            
            landmarks["left_hip"] = (int(landmark_points[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                                   int(landmark_points[mp_pose.PoseLandmark.LEFT_HIP].y * height))
            
            landmarks["right_hip"] = (int(landmark_points[mp_pose.PoseLandmark.RIGHT_HIP].x * width),
                                    int(landmark_points[mp_pose.PoseLandmark.RIGHT_HIP].y * height))
            
            return process_landmarks(landmarks, image, result_image, image_path, output_directory, is_fallback=False)
            
    except Exception as e:
        print(f"MediaPipe processing error: {e}")
        traceback.print_exc()
        return False

def process_with_fallback(image, result_image, image_path, output_directory):
    """Fallback method when pose detection fails or MediaPipe is not available."""
    try:
        height, width, _ = image.shape
        
        # Use simple region detection
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assumed to be the person)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Estimate landmark positions based on bounding box
            landmarks = {
                "nose": (x + w // 2, y + h // 4),
                "left_ear": (x + w // 3, y + h // 4),
                "right_ear": (x + 2 * w // 3, y + h // 4),
                "left_shoulder": (x + w // 3, y + h // 3),
                "right_shoulder": (x + 2 * w // 3, y + h // 3),
                "left_hip": (x + w // 3, y + 2 * h // 3),
                "right_hip": (x + 2 * w // 3, y + 2 * h // 3),
            }
        else:
            # If no contours found, use simple ratios as fallback
            landmarks = {
                "nose": (width // 2, height // 4),
                "left_ear": (width // 2 - width // 10, height // 4),
                "right_ear": (width // 2 + width // 10, height // 4),
                "left_shoulder": (width // 2 - width // 6, height // 3),
                "right_shoulder": (width // 2 + width // 6, height // 3),
                "left_hip": (width // 2 - width // 7, height // 2 + height // 6),
                "right_hip": (width // 2 + width // 7, height // 2 + height // 6),
            }
        
        return process_landmarks(landmarks, image, result_image, image_path, output_directory, is_fallback=True)
        
    except Exception as e:
        print(f"Fallback processing error: {e}")
        traceback.print_exc()
        return False

def process_landmarks(landmarks, image, result_image, image_path, output_directory, is_fallback=False):
    """Process landmarks and generate analysis results."""
    try:
        height, width, _ = image.shape
        
        # Calculate mid-points
        mid_shoulder = (
            (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) // 2,
            (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) // 2
        )
        
        mid_hip = (
            (landmarks["left_hip"][0] + landmarks["right_hip"][0]) // 2,
            (landmarks["left_hip"][1] + landmarks["right_hip"][1]) // 2
        )
        
        mid_ear = (
            (landmarks["left_ear"][0] + landmarks["right_ear"][0]) // 2,
            (landmarks["left_ear"][1] + landmarks["right_ear"][1]) // 2
        )
        
        # Calculate posture metrics
        # 1. Shoulder alignment
        shoulder_angle = abs(90 - abs(math.degrees(math.atan2(
            landmarks["right_shoulder"][1] - landmarks["left_shoulder"][1],
            landmarks["right_shoulder"][0] - landmarks["left_shoulder"][0]
        ))))
        
        # 2. Neck alignment
        neck_angle = calculate_angle(mid_ear, mid_shoulder, mid_hip)
        
        # 3. Hip alignment
        hip_angle = abs(90 - abs(math.degrees(math.atan2(
            landmarks["right_hip"][1] - landmarks["left_hip"][1],
            landmarks["right_hip"][0] - landmarks["left_hip"][0]
        ))))
        
        # 4. Upper body tilt
        upper_body_tilt = abs(90 - abs(math.degrees(math.atan2(
            mid_hip[1] - mid_shoulder[1],
            mid_hip[0] - mid_shoulder[0]
        ))))
        
        # Draw landmarks
        for position in landmarks.values():
            cv2.circle(result_image, position, 5, (0, 255, 0), -1)
        
        # Draw connections
        cv2.line(result_image, landmarks["left_shoulder"], landmarks["right_shoulder"], (0, 255, 0), 2)
        cv2.line(result_image, landmarks["left_shoulder"], landmarks["left_hip"], (0, 255, 0), 2)
        cv2.line(result_image, landmarks["right_shoulder"], landmarks["right_hip"], (0, 255, 0), 2)
        cv2.line(result_image, landmarks["left_hip"], landmarks["right_hip"], (0, 255, 0), 2)
        cv2.line(result_image, landmarks["nose"], mid_shoulder, (0, 255, 0), 2)
        cv2.line(result_image, landmarks["left_ear"], landmarks["left_shoulder"], (0, 255, 0), 2)
        cv2.line(result_image, landmarks["right_ear"], landmarks["right_shoulder"], (0, 255, 0), 2)
        
        # Draw mid-points
        cv2.circle(result_image, mid_shoulder, 5, (255, 0, 0), -1)
        cv2.circle(result_image, mid_hip, 5, (255, 0, 0), -1)
        cv2.circle(result_image, mid_ear, 5, (255, 0, 0), -1)
        
        # Draw vertical reference line
        cv2.line(result_image, (mid_hip[0], mid_hip[1]), (mid_hip[0], 0), (0, 255, 255), 2)
        
        # Add posture feedback text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Thresholds for posture assessment
        shoulder_threshold = 15.0
        neck_threshold = 20.0
        hip_threshold = 10.0
        tilt_threshold = 5.0
        
        # Identify posture issues
        issues = []
        if shoulder_angle > shoulder_threshold:
            issues.append("Uneven shoulders")
        if abs(neck_angle - 180) > neck_threshold:
            issues.append("Forward head")
        if hip_angle > hip_threshold:
            issues.append("Uneven hips")
        if upper_body_tilt > tilt_threshold:
            issues.append("Leaning posture")
        
        # Status text
        status_text = "Posture Status: "
        if issues:
            status_text += ", ".join(issues)
            color = (0, 0, 255)  # Red for issues
        else:
            status_text += "Good"
            color = (0, 255, 0)  # Green for good
        
        # Add text to image with background rectangles for better visibility
        def draw_text_with_background(image, text, position, font, font_scale, text_color, bg_color, thickness):
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            x, y = position
            cv2.rectangle(image, (x, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
            cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

        draw_text_with_background(result_image, status_text, (20, 40), font, 0.7, color, (0, 0, 0), 2)
        draw_text_with_background(result_image, f"Shoulder Angle: {shoulder_angle:.1f}° (Threshold: {shoulder_threshold}°)", 
                                (20, 80), font, 0.6, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_background(result_image, f"Neck Angle: {abs(neck_angle - 180):.1f}° (Threshold: {neck_threshold}°)", 
                                (20, 110), font, 0.6, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_background(result_image, f"Hip Angle: {hip_angle:.1f}° (Threshold: {hip_threshold}°)", 
                                (20, 140), font, 0.6, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_background(result_image, f"Upper Body Tilt: {upper_body_tilt:.1f}° (Threshold: {tilt_threshold}°)", 
                                (20, 170), font, 0.6, (255, 255, 255), (0, 0, 0), 2)
        
        if is_fallback:
            draw_text_with_background(result_image, "Note: Using estimated landmarks", 
                                    (20, height - 30), font, 0.5, (255, 165, 0), (0, 0, 0), 2)
        
        # Generate output path
        file_name = os.path.basename(image_path)
        if output_directory:
            output_path = os.path.join(output_directory, f"analyzed_{file_name}")
        else:
            output_path = os.path.join(os.path.dirname(image_path), f"analyzed_{file_name}")
        
        # Save the result
        cv2.imwrite(output_path, result_image)
        print(f"Analysis saved to {output_path}")
        
        # Display the image
        try:
            cv2.imshow("Posture Analysis", result_image)
            print("Displaying image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image: {e}")
            print("Analysis was saved to file.")
            
        # Print summary
        print("\nPosture Analysis Summary:")
        print(f"Shoulder Angle: {shoulder_angle:.1f}° (Threshold: {shoulder_threshold}°)")
        print(f"Neck Angle: {abs(neck_angle - 180):.1f}° (Threshold: {neck_threshold}°)")
        print(f"Hip Angle: {hip_angle:.1f}° (Threshold: {hip_threshold}°)")
        print(f"Upper Body Tilt: {upper_body_tilt:.1f}° (Threshold: {tilt_threshold}°)")
        
        if issues:
            print(f"Issues Detected: {', '.join(issues)}")
        else:
            print("Posture Status: Good")
            
        return True
        
    except Exception as e:
        print(f"Error in landmark processing: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function with improved menu and image sequence processing."""
    print("Elderly Posture Estimation from Images")
    print("======================================")
    
    # Check for MediaPipe installation
    if MEDIAPIPE_AVAILABLE:
        print("MediaPipe found. Using advanced pose detection.")
    else:
        print("MediaPipe not found. To install, run: pip install mediapipe")
        print("Continuing with basic estimation method...")
    
    # Create output directory variable for reuse
    output_dir = ""
    
    while True:
        print("\nMenu Options:")
        print("1. Analyze an image")
        print("2. Set output directory")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ")
        
        match choice:
            case "1":
                # Get image path
                image_path = input("Enter the path to the image file: ")
                if not image_path:
                    print("No image path provided. Returning to menu.")
                    continue
                
                # Process the image
                analyze_single_image(image_path, output_dir)
                
                # Ask if user wants to analyze another image
                another = input("\nDo you want to analyze another image? (y/n): ").lower()
                if another != 'y':
                    print("Returning to menu.")
            
            case "2":
                # Set output directory
                output_dir = input("Enter output directory path: ")
                if not output_dir:
                    print("No directory provided. Output will be saved alongside input images.")
                else:
                    # Check if directory exists, create if not
                    if not os.path.exists(output_dir):
                        try:
                            os.makedirs(output_dir)
                            print(f"Created output directory: {output_dir}")
                        except Exception as e:
                            print(f"Could not create directory: {e}")
                            output_dir = ""
                    else:
                        print(f"Output directory set to: {output_dir}")
            
            case "3":
                print("Exiting program. Goodbye!")
                break
                
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()