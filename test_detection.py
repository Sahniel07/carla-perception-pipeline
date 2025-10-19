import torch
import requests
import cv2
import os

def run_detection_test():
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    print("Downloading test image...")
    url = "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800"
    try:
        response = requests.get(url, timeout=10)
        # Ensure images folder exists
        images_dir = r"C:\Users\DELL\Desktop\carla-perception-project\images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, "test_car_image.jpg")
        # Save into images folder
        with open(image_path, "wb") as f:
            f.write(response.content)
        print("Test image downloaded to:", image_path)

        print("Running detection and saving YOLOv5 runs output...")
        # This will save auto-saved output under runs/detect
        results = model(image_path)  # no save or save_dir
        # Then manual saving:
        det_images = results.render()
        out_dir = r"C:\Users\DELL\Desktop\carla-perception-project\results"
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(det_images):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(out_dir, f"detected_{idx}.jpg")
            cv2.imwrite(out_path, bgr)
            print("Saved annotated image:", out_path)


        # Print detection summary
        results.print()
        print("Detection test complete! Check the 'results/' folder for output images.")
        return True

    except Exception as e:
        print("Test had issues – that's normal for now.")
        print("Error:", e)
        return False

if __name__ == "__main__":
    print("Testing Object Detection...")
    run_detection_test()
