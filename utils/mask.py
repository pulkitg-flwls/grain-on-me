import cv2
import numpy as np
import mediapipe as mp

def get_face_mask(frame):
    """
    Given an input frame, this function detects the face using MediaPipe's FaceMesh
    and returns a binary mask where the face region is white and the rest is black.
    
    Args:
        frame (numpy.ndarray): Input image (BGR).
    
    Returns:
        numpy.ndarray: Binary mask with the same size as the input frame.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe FaceMesh
    results = face_mesh.process(rgb_frame)
    
    # Create an empty mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get facial landmarks and convert to a NumPy array
            h, w, _ = frame.shape
            points = np.array([
                (int(landmark.x * w), int(landmark.y * h))
                for landmark in face_landmarks.landmark
            ])
            
            # Create a convex hull around the detected face
            hull = cv2.convexHull(points)
            
            # Fill the convex hull in the mask
            cv2.fillConvexPoly(mask, hull, 255)

    face_mesh.close()
    
    return mask

# Example usage:
if __name__ == "__main__":
    # Load an example image
    frame = cv2.imread("noise/denoised.png")
    if frame is None:
        raise ValueError("Image not found!")
    
    # Get the full face mask
    face_mask = get_face_mask(frame)
    # # Get the mouth mask
    # mouth_mask = get_largest_face_mask(frame, mask_type="mouth")
    
    # Display the results
    cv2.imwrite('face_mask.png', (face_mask).astype('uint8'))
    # cv2.imwrite('mouth_mask.png', (mouth_mask).astype('uint8'))