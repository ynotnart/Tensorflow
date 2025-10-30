
# Letter_write_load_v1.00
# draw_infer_gui.py
# This script will load the model. 
#
# Source: ChatGPT
# Created: 10/29/2025
# by Tony Tran

import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf

MODEL_KERAS = "mnist_cnn_model.keras"

# 1) Load model
def load_model():
    if os.path.exists(MODEL_KERAS):
        print(f"Loading model from '{MODEL_KERAS}'...")
        return tf.keras.models.load_model(MODEL_KERAS)
    else:
        raise FileNotFoundError(
            "Model not found. Run 'train_save_mnist.py' first to create the model file."
        )


model = load_model()
model.summary()  # optional: prints model architecture to console

# 2) Tkinter GUI
class DrawDigits(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit (0-9) — Predict")
        self.geometry("320x380")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        button_frame = tk.Frame(self)
        button_frame.pack(fill='x')

        self.pred_label = tk.Label(button_frame, text="Draw and press Predict", font=("Helvetica", 12))
        self.pred_label.pack(side='left', padx=8)

        self.button_predict = tk.Button(button_frame, text="Predict", command=self.predict_digit)
        self.button_predict.pack(side='right', padx=6)
        self.button_clear = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(side='right', padx=6)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image1 = Image.new("L", (280,280), color=255)  # white background
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        x1, y1 = (event.x-8), (event.y-8)
        x2, y2 = (event.x+8), (event.y+8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,280,280], fill=255)
        self.pred_label.config(text="Draw and press Predict")

    def predict_digit(self):
        # Resize to 28x28 & invert colors if needed (MNIST is white background / black digit)
        img = self.image1.resize((28,28))
        img_arr = np.array(img).astype(np.float32) / 255.0  # shape (28,28)
        # if your drawings are inverted, uncomment:
        # img_arr = 1.0 - img_arr

        # Prepare for model
        img_arr = img_arr[np.newaxis, ..., np.newaxis]  # (1,28,28,1)
        preds = model.predict(img_arr)
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds))

        self.pred_label.config(text=f"Predicted: {digit} (conf {confidence:.2f})")

if __name__ == "__main__":
    app = DrawDigits()
    app.mainloop()



