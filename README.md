# Smart-Warehouse-Optimization-using-AI
Objective:
The goal of this project is to build a smart warehouse AI system that can automatically detect and localize packages, shelves, barcodes, and other storage items using computer vision. This capability is foundational for automating inventory audits, robotics integration, and optimizing item movement in logistics environments.
To train a YOLOv8 object detection model using Roboflow, we begin by sourcing or creating a well-annotated dataset. Roboflow provides a robust platform to either upload a custom dataset or fork an existing public dataset from Roboflow Universe. Each image in the dataset is labeled with bounding boxes and class names, and the dataset can be exported in various formats. For YOLOv8, the YOLOv5 export format is used, as YOLOv8 is fully compatible with YOLOv5-style annotations. After selecting or creating the dataset, users can generate a version and choose the "YOLOv5" format to download it. Roboflow then provides a Python code snippet containing the user's API key, workspace name, project name, and version number. This snippet is executed within a Google Colab environment to programmatically download the dataset. Once downloaded, the dataset contains a data.yaml configuration file along with the train/ and valid/ directories. The YOLOv8 model is loaded (e.g., yolov8n.pt or yolov8s.pt) using the Ultralytics library, and the training process is initiated by referencing the data.yaml file. This pipeline automates data preprocessing, annotation management, and dataset integration, significantly accelerating the model development cycle for object detection tasks.

<img width="746" height="714" alt="image" src="https://github.com/user-attachments/assets/f8e1fda6-e947-41fb-92e9-f36cb09642eb" />

<img width="1032" height="440" alt="image" src="https://github.com/user-attachments/assets/bdb82272-5ca4-427c-b401-350f61572565" />

<img width="634" height="404" alt="image" src="https://github.com/user-attachments/assets/86901e14-130d-47f0-ad12-f01030551090" />

Model Used:
YOLOv8 (You Only Look Once v8): A state-of-the-art object detection model by Ultralytics that supports real-time detection and high accuracy in constrained environments like warehouses.

Tools & Platforms:
Google Colab: For model development and training in the cloud

Roboflow: For dataset annotation, preprocessing, and exporting in YOLO format

Ultralytics Library: To use YOLOv8 models

OpenCV/Matplotlib: For result visualization

Workflow: YOLOv8 Training Using Roboflow
✅ Step 1: Dataset Creation or Selection
The dataset must contain images of packages, shelves, barcodes, or any relevant warehouse items.

You can either:

Upload your own custom dataset to Roboflow

Or search and fork an existing public dataset from Roboflow Universe

Step 2: Labeling and Annotation
Use Roboflow’s annotation tool to manually draw bounding boxes and assign class labels to each object.

Ensure consistency in labeling (e.g., classes like package, shelf, box).

Once done, generate a version of the dataset.

Step 3: Export Dataset in YOLOv5 Format
YOLOv8 supports YOLOv5-style annotations, so choose YOLOv5 format during export.

Roboflow will generate a downloadable Python snippet, which includes:

api_key

workspace

project name

version number

Step 3: Export Dataset in YOLOv5 Format
YOLOv8 supports YOLOv5-style annotations, so choose YOLOv5 format during export.

Roboflow will generate a downloadable Python snippet, which includes:

api_key

workspace

project name

version number

/content/your-dataset-name/
├── train/
├── valid/
├── data.yaml
!pip install ultralytics --quiet
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt
model.train(
    data="/content/your-dataset-name/data.yaml",
    epochs=25,
    imgsz=640
)
/content/runs/detect/train/
from google.colab import drive
drive.mount('/content/drive')

!cp /content/runs/detect/train/weights/best.pt /content/drive/MyDrive/warehouse_yolov8_best.pt
results = model.predict("test_image.jpg")
results[0].save(filename="output.jpg")

# Display result
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("output.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
| Feature                         | Description                                  |
| ------------------------------- | -------------------------------------------- |
| **No Manual Format Conversion** | Roboflow exports in YOLO-ready format        |
| **Web-Based Annotation Tool**   | Easy bounding box drawing and class tagging  |
| **API Integration**             | One-line dataset download into Colab         |
| **Cloud-Based Training**        | No need for local setup; all in Google Colab |
This integration of Roboflow for data handling and YOLOv8 for training significantly speeds up the development of custom object detection systems. In smart warehouses, such models can be deployed on robots, drones, or cameras to automate real-time inventory management, package scanning, or object localization — reducing human labor and increasing operational accuracy.


