import socket
import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# --- Configuration ---
SAVE_DIR = r"D:\VIII sem\Project\Dataset\new bike images"  # Save path for received images
OUTPUT_FOLDER = r"D:\VIII sem\Project\Dataset\new bike images\output" # Path for processed YOLO output
MODEL_PATH = r"D:\VIII sem\Project\Model\Helmet detection\Hemet detection\best (5).pt"  # Your trained YOLOv8 model

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Get number of existing output images for naming
existing_files = os.listdir(OUTPUT_FOLDER)
image_count = len([f for f in existing_files if f.endswith(".jpg")])

# Create server socket
server_socket = socket.socket()
server_socket.bind(("0.0.0.0", 5001))
server_socket.listen(1)

print("🚦 Waiting for image from Raspberry Pi...")

while True:
    conn, addr = server_socket.accept()
    print(f"🔗 Connection from {addr}")

    # Save received image
    filename = f"received_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "wb") as f:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            f.write(data)
    conn.close()
    print(f"📸 Image saved as {filepath}")

    # Run YOLO inference
    results = model(filepath, save=True)

    # Locate latest YOLO output folder (e.g., "predict", "predict1", ...)
    latest_folder = max(
        [d for d in os.listdir("runs/detect/") if "predict" in d],
        key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
    )

    latest_output_folder = os.path.join("runs/detect", latest_folder)
    output_images = [f for f in os.listdir(latest_output_folder) if f.endswith(".jpg")]

    # Move and rename YOLO output images
    for img_name in output_images:
        src_path = os.path.join(latest_output_folder, img_name)
        new_img_name = f"predict{image_count + 1}.jpg"
        dest_path = os.path.join(OUTPUT_FOLDER, new_img_name)
        os.rename(src_path, dest_path)
        image_count += 1

        # Display or log output
        print(f"✅ Processed image saved: {dest_path}")

        # Optionally display
        img = cv2.imread(dest_path)
        # cv2.imshow("Detection", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
