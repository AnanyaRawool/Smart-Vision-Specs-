import cv2
import pytesseract
from PIL import Image
import numpy as np
import time

# Path to Tesseract-OCR (Change this if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set the camera index (0 for default camera)
CAMERA_INDEX = 0  

def live_ocr():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Use DirectShow for Windows users

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Convert frame to grayscale for better OCR accuracy
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Capture an image every 5 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            last_capture_time = current_time  # Reset timer

            # Convert OpenCV image to PIL image
            pil_image = Image.fromarray(gray_frame)

            # Perform OCR
            extracted_text = pytesseract.image_to_string(pil_image)

            # Display extracted text
            print("\nDetected Text:\n", extracted_text.strip())

        # Show live video feed
        cv2.imshow("Live OCR - Press 'q' to exit", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the live OCR function
live_ocr()

