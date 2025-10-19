# run_simple_test.py

from simple_detection import load_detection_model, process_image
import cv2
import os

def main():
    # Load the YOLOv5 model
    print("Loading detection model...")
    model = load_detection_model()

    # Read the test image you downloaded earlier
    image_path = r"C:\Users\DELL\Desktop\carla-perception-project\images\test_car_image.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'.")
        print("Please download the test image by running: python test_detection.py")
        return

    # Run object detection
    print("Running object detection on test image...")
    results = process_image(img, model)

    # Render detections onto images
    det_images = results.render()  # det_images is a list of numpy arrays

    # Save each rendered image to your results folder

    out_dir = r"C:\Users\DELL\Desktop\carla-perception-project\results"
    os.makedirs(out_dir, exist_ok=True)

    for idx, img in enumerate(det_images):
        # img is RGB, convert to BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, f"detected_{idx}.jpg")
        cv2.imwrite(out_path, bgr)
    print(f"Saved: {out_path}")
    print("Detection complete. Check the 'results/' folder for output images.")

if __name__ == "__main__":
    main()
