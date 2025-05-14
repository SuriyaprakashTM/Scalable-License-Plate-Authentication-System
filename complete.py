import os
import cv2
import torch
import time
import yaml
import socket
import requests
import tkinter as tk
from tkinter import simpledialog
from shutil import copyfile
from ultralytics import YOLO
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import gspread
import matplotlib.pyplot as plt

# --------------------------- CONFIGURATION --------------------------- #
CONFIG = {
    "source_folder": r"D:\Telegram",
    "destination_folder": r"D:\VIII sem\Project\Dataset\New car images",
    "model_path": r"D:\VIII sem\Project\Model\autoretrain\final_01\best_01.pt",
    "seatbelt_model_path": r"D:\VIII sem\Project\Model\seatbelt\new model\best.pt",
    "data_yaml_path": r"D:\VIII sem\Project\Model\autoretrain\final_01\data_1.yaml",
    "training_runs_path": r"D:\VIII sem\Project\Model\training_runs",
    "plate_recognizer_api_key": "af09a58183c51e47df8719168a189b279acbf4f4",
    "service_account_file": "credentials.json",
    "drive_folder_id": "1SlzL-DF0z5CYceHBld4P8Fx83A1DfyHv",
    "spreadsheet_id": "1-SxXHJmZ4dePNpf8eHhC4PS5e_lm8xT0L1e8NrWZRGw",
    "image_threshold": 5,
    "tracking_file": os.path.join(r"D:\VIII sem\Project\Dataset\New car images", "last_image_count.txt"),
    "socket_port": 5001,
    "confidence_threshold": 0.5  # <<< NEWLY ADDED
}

torch.cuda.empty_cache()
car_detector_model = YOLO("yolov8n.pt")
make_model = YOLO(CONFIG["model_path"]) if os.path.exists(CONFIG["model_path"]) else None
seatbelt_model = YOLO(CONFIG["seatbelt_model_path"])

# ----------------------- GOOGLE SERVICES INIT ------------------------ #
def init_google_services():
    creds = service_account.Credentials.from_service_account_file(
        CONFIG["service_account_file"],
        scopes=["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/spreadsheets"]
    )
    drive_service = build("drive", "v3", credentials=creds)
    sheets_client = gspread.authorize(creds)
    return drive_service, sheets_client

drive_service, sheets_client = init_google_services()

# --------------------------- IMAGE PROCESS ---------------------------- #
def detect_and_crop_car(image_path):
    image = cv2.imread(image_path)
    results = car_detector_model(image_path)
    conf_thresh = CONFIG["confidence_threshold"]

    for result in results:
        for box in result.boxes:
            if box.conf[0] >= conf_thresh:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = image[y1:y2, x1:x2]
                cropped_path = os.path.join(CONFIG["destination_folder"], "cropped_car.jpg")
                cv2.imwrite(cropped_path, cropped)
                plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
                return cropped_path
    return None


def detect_car_make(image_path):
    if not make_model:
        return "Unknown"

    results = make_model(image_path)
    image = cv2.imread(image_path)
    detected_makes = []
    conf_thresh = CONFIG["confidence_threshold"]

    for result in results:
        for box in result.boxes:
            if box.conf[0] >= conf_thresh:
                cls_id = int(box.cls[0].item())
                class_name = make_model.names[cls_id]
                detected_makes.append(class_name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Detected", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return detected_makes[0] if detected_makes else "Unknown"


def detect_seat_belt(image_path):
    results = seatbelt_model(image_path)
    conf_thresh = CONFIG["confidence_threshold"]

    labels = []
    for r in results:
        for box in r.boxes:
            if box.conf[0] >= conf_thresh:
                label = seatbelt_model.names[int(box.cls[0])]
                labels.append(label)

    status = "Seatbelt Detected" if any("seatbelt" in l.lower() for l in labels) else "No Seatbelt"
    print("Seatbelt Status:", status)
    return status


def get_car_model_name():
    root = tk.Tk()
    root.withdraw()
    return simpledialog.askstring("Input", "Enter the car model name:")

def get_license_plate(image_path):
    url = "https://api.platerecognizer.com/v1/plate-reader/"
    headers = {"Authorization": f"Token {CONFIG['plate_recognizer_api_key']}"}
    try:
        with open(image_path, "rb") as img:
            response = requests.post(url, headers=headers, files={"upload": img})
            response.raise_for_status()
            plate_data = response.json()
            if plate_data.get("results"):
                return plate_data["results"][0]["plate"].upper()
            else:
                print("No plate found in response:", plate_data)
    except Exception as e:
        print("Plate recognition error:", str(e))
    return "NOT DETECTED"

def upload_to_drive(image_path, car_name, plate):
    metadata = {"name": f"{car_name}_{plate}.jpg", "parents": [CONFIG["drive_folder_id"]]}
    media = MediaFileUpload(image_path, mimetype="image/jpeg")
    file = drive_service.files().create(body=metadata, media_body=media, fields="id").execute()
    return f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"

def update_google_sheet(car_name, plate, link, seatbelt_status):
    sheet = sheets_client.open_by_key(CONFIG["spreadsheet_id"]).sheet1
    sheet.append_row([car_name, plate, seatbelt_status, link, time.strftime('%Y-%m-%d %H:%M:%S')])

def update_data_yaml_if_needed(name):
    with open(CONFIG["data_yaml_path"], "r") as f:
        data = yaml.safe_load(f)

    if name not in data["names"]:
        data["names"].append(name)
        data["nc"] = len(data["names"])
        with open(CONFIG["data_yaml_path"], "w") as f:
            yaml.dump(data, f, sort_keys=False)

def auto_retrain_model():
    current_images = [f for f in os.listdir(CONFIG["destination_folder"]) if f.lower().endswith((".jpg", ".png"))]
    current_count = len(current_images)
    last_count = int(open(CONFIG["tracking_file"]).read().strip()) if os.path.exists(CONFIG["tracking_file"]) else 0

    if current_count - last_count >= CONFIG["image_threshold"]:
        from roboflow import Roboflow
        rf = Roboflow(api_key="9wrVLLe526g5NHMuu8F0")
        project = rf.workspace("cdp-dlm3v").project("car-features-pmivz")
        version = project.version(3)
        version.download("yolov8")

        make_model.train(
            data=CONFIG["data_yaml_path"],
            device="cpu",
            imgsz=416,
            epochs=15,
            batch=2,
            resume=False,
            project=CONFIG["training_runs_path"]
        )
        make_model.export(format="torchscript", save_dir=CONFIG["model_path"])

        with open(CONFIG["tracking_file"], "w") as f:
            f.write(str(current_count))

def process_new_image(image_path):
    cropped = detect_and_crop_car(image_path)
    if not cropped:
        print("No car detected.")
        return

    detected_make = detect_car_make(cropped)
    seatbelt_status = detect_seat_belt(cropped)

    with open(CONFIG["data_yaml_path"], "r") as file:
        data_yaml = yaml.safe_load(file)
    existing_names = data_yaml.get("names", [])

    if detected_make in existing_names:
        full_car_name = detected_make
    else:
        model_name = get_car_model_name()
        if not model_name:
            return
        full_car_name = f"{detected_make} {model_name}".strip()
        update_data_yaml_if_needed(full_car_name)

    plate = get_license_plate(cropped)
    drive_link = upload_to_drive(cropped, full_car_name, plate)
    update_google_sheet(full_car_name, plate, drive_link, seatbelt_status)

    new_path = os.path.join(CONFIG["destination_folder"], f"{full_car_name.replace(' ', '_')}_{plate}.jpg")
    copyfile(cropped, new_path)

    print(f"Processed: {full_car_name}, Plate: {plate}, Seatbelt: {seatbelt_status}, Link: {drive_link}")
    auto_retrain_model()

# --------------------------- SOCKET SERVER --------------------------- #
def receive_images_from_pi():
    server_socket = socket.socket()
    server_socket.bind(("0.0.0.0", CONFIG["socket_port"]))
    server_socket.listen(1)
    print("Waiting for image from Raspberry Pi...")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")
        filename = f"received_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(CONFIG["source_folder"], filename)
        
        with open(filepath, "wb") as f:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                f.write(data)
        conn.close()
        print(f"Image saved as {filename}")
        process_new_image(filepath)

# ----------------------------- MAIN ---------------------------------- #
if __name__ == "__main__":
    os.makedirs(CONFIG["source_folder"], exist_ok=True)
    os.makedirs(CONFIG["destination_folder"], exist_ok=True)
    receive_images_from_pi()



