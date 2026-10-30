# Indian Currency Note Recognition System

A deep learning project that recognizes Indian rupee notes (Rs. 10, 20, 50, 100, 200, 500, 2000) from an uploaded image or a live webcam feed. The result is shown as text in a desktop window ("This is a 50 rupees note") and can also be spoken aloud.

## 1. Project Overview
- **Problem:** Identify the denomination of an Indian currency note automatically from an image.
- **Approach:** Transfer learning with MobileNetV2 (pretrained on ImageNet), fine-tuned on a labeled currency dataset.
- **Output:** Denomination text + confidence percentage + optional voice.
- **Use cases:** Assisting visually impaired people, cash-counting machines, kiosks, learning projects.

## 2. Folder Structure
```
Indian_Currency_Recognition/
|-- dataset/
|   |-- train/
|   |   |-- 10/  20/  50/  100/  200/  500/  2000/  background/
|   |-- val/
|       |-- 10/  20/  50/  100/  200/  500/  2000/  background/
|-- train_model.py      (trains the model)
|-- currency_app.py     (desktop app)
|-- requirements.txt    (libraries)
|-- README.md           (this file)

Created after training:
|-- currency_model.h5   (trained model)
|-- labels.txt          (class names)
```

## 3. Dataset
Download the ready-made, already-labeled dataset from Kaggle:

**Indian Currency Note images dataset 2020 (Vishal Mane)**
https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020

It has 8 classes: 10, 20, 50, 100, 200, 500, 2000 rupees and background. The folder name is the label, so no manual labeling is needed.

Other options:
- Indian Currency Notes Classifier (Gaurav Sahani): https://www.kaggle.com/datasets/gauravsahani/indian-currency-notes-classifier
- Roboflow Universe "indian currency notes" (Omkar Patkar), for object detection with bounding boxes.

**Where to put it:** unzip and copy the class folders into `dataset/train/`.
- If the dataset has its own train and val folders, copy them to `dataset/train/` and `dataset/val/`.
- If it has only one set of folders, put them all in `dataset/train/` and leave `dataset/val/` empty. The script automatically splits 80% training and 20% validation.
- Delete the `PUT_IMAGES_HERE.txt` placeholder files if you like (they are ignored).
- Folder names must be plain numbers like `50`, so the app says "This is a 50 rupees note".

## 4. Installation
1. Install Python 3.9 to 3.11 from https://www.python.org
2. Open a terminal in the project folder and run:
```
pip install -r requirements.txt
```
On Linux, also run `sudo apt install python3-tk`.

## 5. How to Run
```
python train_model.py     # trains the model, takes about 10-30 minutes
python currency_app.py    # opens the window app
```
In the app, click **Upload Image** or **Open Camera**, then show a note. Hold the note flat, well lit, filling most of the frame.

## 6. Tools and Technologies Used (A to Z)
| Tool | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Building and training the deep learning model |
| MobileNetV2 | Lightweight pretrained CNN used for transfer learning |
| ImageNet weights | Pretrained knowledge so training needs less data and time |
| OpenCV (cv2) | Reading images, resizing, webcam capture, color conversion |
| NumPy | Array and numerical operations |
| Pillow (PIL) | Converting images for display in the window |
| Tkinter | Building the desktop window, buttons and labels |
| pyttsx3 | Offline text-to-speech (speaks the result) |
| Kaggle dataset | Labeled training images |

## 7. How It Works
1. **Data loading:** Images are read from folders; folder names become class labels.
2. **Augmentation:** Random flip, rotation, zoom and contrast change make the model robust to different angles and lighting.
3. **Preprocessing:** Images are resized to 224x224 and scaled for MobileNetV2 (this step is built into the model).
4. **Model:** MobileNetV2 base (feature extractor) -> Global Average Pooling -> Dropout (0.3) -> Dense softmax layer with one output per class.
5. **Training Stage 1:** Base frozen, only the new top layers are trained (8 epochs).
6. **Training Stage 2 (fine-tuning):** The last 30 layers of MobileNetV2 are unfrozen and trained with a small learning rate (6 epochs) for higher accuracy.
7. **Saving:** Model saved as `currency_model.h5`, class names as `labels.txt`.
8. **Prediction:** The app resizes the image, runs the model, takes the highest-probability class, and shows it as text. Results below 60% confidence, or the "background" class, show "No clear currency note detected".
9. **Camera mode:** A frame is analyzed every 0.3 seconds. The voice speaks only when the answer changes.

## 8. Troubleshooting
- **Wrong labels like "50rs":** rename dataset folders to plain numbers, or edit `result_text()` in `currency_app.py`.
- **Poor camera accuracy:** better lighting, plain background, note filling the frame. Add your own phone photos to the training folders and retrain.
- **Low accuracy overall:** increase the epochs in `train_model.py`.
- **No voice:** voice is optional. If pyttsx3 fails, text output still works.
- **Camera not opening:** close other apps using the camera, or change `VideoCapture(0)` to `VideoCapture(1)`.

## 9. Future Improvements
- Detect multiple notes in one image (object detection with YOLO).
- Fake note detection.
- Convert to TensorFlow Lite for a mobile app.
- Add coin recognition and total amount calculation.

## 10. Short Summary (for internship report)
The system classifies Indian currency notes of 10, 20, 50, 100, 200, 500 and 2000 rupees from images or live camera input. It uses transfer learning with MobileNetV2 trained on a labeled Kaggle dataset, with data augmentation to handle different lighting and angles. A Tkinter desktop interface displays the detected denomination as text and speaks it aloud. Tools: Python, TensorFlow/Keras, OpenCV, Tkinter, Pillow, pyttsx3.
