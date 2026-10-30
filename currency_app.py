"""
Desktop app: detect Indian currency notes from an image or live camera.
Shows the result as text and speaks it aloud (optional).
Run:  python currency_app.py   (after training)
"""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

try:
    import pyttsx3
    VOICE = True
except ImportError:
    VOICE = False

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60

model = tf.keras.models.load_model("currency_model.h5")
with open("labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]


def speak(text):
    if not VOICE:
        return
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def predict(rgb_image):
    """rgb_image: numpy array (RGB). Returns (label, confidence).
    The model already contains preprocessing, so we pass raw 0-255 pixels."""
    img = cv2.resize(rgb_image, IMG_SIZE).astype("float32")
    img = np.expand_dims(img, axis=0)
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def result_text(label, conf):
    if label.lower() == "background" or conf < CONF_THRESHOLD:
        return "No clear currency note detected. Try a clearer image."
    return f"This is a {label} rupees note  ({conf * 100:.1f}% sure)"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Indian Currency Note Detector")
        root.geometry("720x660")
        self.camera_on = False
        self.cap = None
        self.last_spoken = ""

        tk.Label(root, text="Indian Currency Note Detector",
                 font=("Arial", 18, "bold")).pack(pady=10)

        self.panel = tk.Label(root, bg="gray85", width=80, height=24)
        self.panel.pack(pady=5)

        self.result = tk.Label(root, text="Upload an image or open the camera",
                               font=("Arial", 16, "bold"), fg="darkgreen",
                               wraplength=680)
        self.result.pack(pady=10)

        row = tk.Frame(root)
        row.pack(pady=5)
        tk.Button(row, text="Upload Image", font=("Arial", 12),
                  command=self.upload).grid(row=0, column=0, padx=10)
        self.cam_btn = tk.Button(row, text="Open Camera", font=("Arial", 12),
                                 command=self.toggle_camera)
        self.cam_btn.grid(row=0, column=1, padx=10)
        tk.Button(row, text="Exit", font=("Arial", 12),
                  command=self.close).grid(row=0, column=2, padx=10)

        root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self, rgb):
        img = Image.fromarray(rgb)
        img.thumbnail((640, 440))
        photo = ImageTk.PhotoImage(img)
        self.panel.configure(image=photo, width=photo.width(), height=photo.height())
        self.panel.image = photo

    def upload(self):
        self.stop_camera()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", "Could not read that image.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.show(rgb)
        label, conf = predict(rgb)
        text = result_text(label, conf)
        self.result.config(text=text)
        speak(text)

    def toggle_camera(self):
        if self.camera_on:
            self.stop_camera()
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the camera.")
            return
        self.camera_on = True
        self.cam_btn.config(text="Stop Camera")
        self.update_frame()

    def update_frame(self):
        if not self.camera_on:
            return
        ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, conf = predict(rgb)
            text = result_text(label, conf)
            self.result.config(text=text)
            self.show(rgb)
            if (text != self.last_spoken and conf >= CONF_THRESHOLD
                    and label.lower() != "background"):
                self.last_spoken = text
                speak(text)
        self.root.after(300, self.update_frame)

    def stop_camera(self):
        self.camera_on = False
        self.cam_btn.config(text="Open Camera")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
