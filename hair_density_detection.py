import tkinter as tk
import cv2
import numpy as np
from tkinter import messagebox



def calculate_salt11(prediction, grid_size=10, threshold=0.5):
    total_hairless_grids = 0
    total_grids = 0

    height, width = prediction.shape[:2]
    h_step, w_step = height // grid_size, width // grid_size

    for i in range(0, height, h_step):
        for j in range(0, width, w_step):
            grid_prediction = prediction[i: i + h_step, j: j + w_step]
            if np.mean(grid_prediction) < threshold:
                total_hairless_grids += 1
            total_grids += 1

    salt11_score = (total_hairless_grids / total_grids) * 100
    return salt11_score



def initialize_camera():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
    return cap



def capture_image(side):
    side_names = ["front", "back", "left", "right"]
    cap = initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            break
        cv2.imshow(f"Capture Image - {side_names[side]}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to capture image
            cv2.imwrite(f"captured_image_{side_names[side]}.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame



def process_image(side):
    side_names = ["front", "back", "left", "right"]
    frame = capture_image(side)

    if frame is None:
        return

    # Preprocess frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (100, 100))  # Resize frame to match model input size
    frame = frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Make prediction using your model (replace this with actual model inference)
    prediction = np.random.rand(100, 100)  # Placeholder prediction

    # Calculate SALT 11 score
    salt11_score = calculate_salt11(prediction)

    # Display SALT 11 score in a messagebox
    messagebox.showinfo(f"SALT 11 Score - {side_names[side]}", f"SALT 11 Score: {salt11_score:.2f}")


# Create the UI using tkinter
def create_ui():
    root = tk.Tk()
    root.title("SALT 11 Score Calculator")

    # Set window size and background color
    root.geometry("400x300")
    root.configure(bg='#f0f0f0')

    # Create buttons for different sides
    tk.Label(root, text="Capture Image and Calculate SALT 11 Score", font=("Helvetica", 14), bg='#f0f0f0').pack(pady=20)

    tk.Button(root, text="Capture Front Image", command=lambda: process_image(0), font=("Helvetica", 12), bg='#5c85d6', fg='white').pack(pady=10)
    tk.Button(root, text="Capture Back Image", command=lambda: process_image(1), font=("Helvetica", 12), bg='#5c85d6', fg='white').pack(pady=10)
    tk.Button(root, text="Capture Left Image", command=lambda: process_image(2), font=("Helvetica", 12), bg='#5c85d6', fg='white').pack(pady=10)
    tk.Button(root, text="Capture Right Image", command=lambda: process_image(3), font=("Helvetica", 12), bg='#5c85d6', fg='white').pack(pady=10)

    # Start the tkinter main loop
    root.mainloop()


if __name__ == "__main__":
    create_ui()
