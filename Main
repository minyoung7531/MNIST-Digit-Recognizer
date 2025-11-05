# Project Title: MNIST Digit Recognizer
# By Daniel (Minyoung) Park
# 26 May 2025
# Outside Sources Used ...
    # 1. https://www.geeksforgeeks.org/mnist-dataset/
    # 2. https://www.ibm.com/think/topics/machine-learning
    # 3. https://lakefs.io/blog/data-preprocessing-in-machine-learning/
    # 4. https://www.purestorage.com/knowledge/what-is-data-preprocessing.html
    # 5. https://www.youtube.com/watch?v=ukzFI9rgwfU
    # 6. https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/
    # 7. https://www.geeksforgeeks.org/learning-model-building-scikit-learn-python-machine-learning-library/
    # 8. https://stackoverflow.com/questions/43577665/deskew-mnist-images
    # 9. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # 10. https://scikit-learn.org/stable/modules/calibration.html

import ssl  # Standard library for TLS/SSL communication
ssl._create_default_https_context = ssl._create_unverified_context  # Disable SSL certificate verification globally for this session

import os  # Operating system interfaces
import tkinter as tk  # GUI toolkit for Python
from tkinter import filedialog, Label, Button, Scale, messagebox  # Specific GUI components
from PIL import Image, ImageTk, ImageDraw  # PIL for image operations and drawing
import numpy as np  # Numerical computing library
import cv2  # OpenCV for image processing operations
from sklearn.datasets import fetch_openml  # Function to fetch datasets from OpenML
from sklearn.model_selection import train_test_split  # Utility to split data into train/validation sets
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron classifier implementation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Metrics for evaluating classifier
import joblib  # Library for saving and loading Python objects
import matplotlib.pyplot as plt  # Plotting library for visualization

# Function to deskew an image based on its moments
def deskew(img: np.ndarray) -> np.ndarray:
    m = cv2.moments(img)  # Calculate image moments for skew detection
    if abs(m['mu02']) < 1e-2:  # If vertical moment is negligible, skip deskew
        return img.copy()  # Return a copy of the original image
    skew = m['mu11'] / m['mu02']  # Compute skew based on moments
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew],  # Affine transform matrix for deskew
                    [0,     1,                       0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),  # Apply warp with inverse mapping for interpolation
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

# Function to center the digit in the image by shifting its centroid
def center_image(img: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # Convert to binary image
    m = cv2.moments(bw)  # Compute moments on binary image
    if m['m00'] == 0:  # If no foreground pixels, return original
        return img  # Return original image if empty
    cx = m['m10'] / m['m00']  # Centroid x-coordinate
    cy = m['m01'] / m['m00']  # Centroid y-coordinate
    rows, cols = img.shape  # Get image dimensions
    shiftx = int(round(cols/2.0 - cx))  # Compute horizontal shift amount
    shifty = int(round(rows/2.0 - cy))  # Compute vertical shift amount
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])  # Translation matrix
    return cv2.warpAffine(img, M, (cols, rows))  # Apply translation to center image

# Preprocess an input image or file path for MNIST classification
def preprocess_opencv(input_img):
    if isinstance(input_img, str):  # If input is a file path
        gray = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    else:
        gray = input_img.copy()  # Otherwise, copy the array

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    _, th = cv2.threshold(blur, 0, 255,  # Otsu's thresholding and invert image
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find external contours

    if not cnts:  # If no contours found
        return np.zeros((1, 784), dtype=float)  # Return zero array placeholder

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))  # Get bounding box of largest contour
    img_h, img_w = gray.shape  # Original image size
    border_tol = 5  # Tolerance for border artifact detection

    # Detect if the digit touches the image border
    if x <= border_tol or y <= border_tol or x + w >= img_w - border_tol or y + h >= img_h - border_tol:
        keep = messagebox.askyesno(
            "Border Artifact Detected",  # Title of the dialog
            "Your digit touches the edge of the image, which may affect the prediction.\n"
            "Would you like to remove edge artifacts and retry?"  # Message text
        )
        explain = messagebox.askyesno(  # Ask if they want an explanation of why
            "Why this warning?",  # Title of the explanation prompt
            "Digits at the very edge can skew centering & feature-extraction.\n"
            "Would you like to know more about how this impacts MNIST accuracy?"  # Prompt text
        )
        if explain:  # If user wants to know why
            messagebox.showinfo(  # Show the detailed explanation
                "Why Edge-Artifacts Matter",  # Explanation title
                "MNIST models assume digits are roughly centered and scaled.\n"  # Explanation text line 1
                "Pixels cut off at the border shift the center of mass\n"  # Explanation text line 2
                "and distort stroke-based features, so the classifier\n"  # Explanation text line 3
                "may latch onto the artifact instead of the true shape."  # Explanation text line 4
            )
        if keep:  # If user agreed to remove artifacts
            th[:border_tol, :] = 0  # Remove top border pixels
            th[-border_tol:, :] = 0  # Remove bottom border pixels
            th[:, :border_tol] = 0  # Remove left border pixels
            th[:, -border_tol:] = 0  # Remove right border pixels
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Re-find contours
            if not cnts:  # If still none
                return np.zeros((1, 784), dtype=float)  # Return zero array
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))  # Update bounding box

    digit = th[y:y+h, x:x+w]  # Crop the region of interest
    max_size = max(h, w)  # Determine max dimension for scaling
    ratio = 20.0 / max_size  # Scaling factor to fit 20x20 area
    new_h, new_w = int(h * ratio), int(w * ratio)  # Calculate new dimensions

    if new_h == 0 or new_w == 0:  # Guard against zero-sized outputs
        return np.zeros((1, 784), dtype=float)  # Return zero array

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize digit
    canvas_20 = np.zeros((20, 20), dtype=np.uint8)  # Create blank 20x20 canvas
    off_y = (20 - new_h) // 2  # Vertical offset for centering
    off_x = (20 - new_w) // 2  # Horizontal offset for centering
    canvas_20[off_y:off_y+new_h, off_x:off_x+new_w] = resized  # Paste resized digit

    canvas_28 = np.zeros((28, 28), dtype=np.uint8)  # Create blank 28x28 canvas
    off = (28 - 20) // 2  # Offset to center 20x20 in 28x28
    canvas_28[off:off+20, off:off+20] = canvas_20  # Place 20x20 digit in center
    deskewed = deskew(canvas_28)  # Deskew image
    centered = center_image(deskewed)  # Center the deskewed image

    return (centered.flatten() / 255.0).reshape(1, -1)  # Normalize and reshape to 1x784 array

MODEL_PATH = 'mnist_mlp.pkl'  # Path to save/load trained model data
# Load existing model or create and train a new one if not present
if os.path.exists(MODEL_PATH):
    data = joblib.load(MODEL_PATH)  # Load saved classifier and validation set
    clf    = data['clf']  # Extract classifier instance
    X_val  = data['X_val']  # Validation data features
    y_val  = data['y_val']  # Validation data labels
else:
    print("Fetching MNIST…")  # Notify dataset download
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)  # Fetch dataset
    X      = X / 255.0  # Normalize pixel values
    y      = y.astype(int)  # Convert labels to integers
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)  # Split into train and validation sets
    print("Training MLP…")  # Notify training start
    clf = MLPClassifier(
        hidden_layer_sizes=(150, 100),  # Two hidden layers (150 and 100 neurons)
        activation='relu',  # ReLU activation function
        solver='adam',  # Adam optimizer
        alpha=1e-4,  # L2 regularization strength
        batch_size=256,  # Size of minibatches
        learning_rate_init=0.001,  # Initial learning rate
        max_iter=60,  # Maximum training iterations
        early_stopping=True,  # Enable early stopping based on validation score
        validation_fraction=0.1,  # Fraction of training data used for validation
        verbose=True,  # Print training progress
        random_state=42  # Random seed for reproducibility
    )
    clf.fit(X_train, y_train)  # Train MLP on training data
    joblib.dump({'clf': clf, 'X_val': X_val, 'y_val': y_val}, MODEL_PATH)  # Save trained model and validation data
    print("Validation accuracy:", clf.score(X_val, y_val))  # Print validation score
print("Ready.")  # Indicate readiness to the user

# Initialize main GUI window
root = tk.Tk()  # Create root application window
root.title("MNIST Digit Recognizer")  # Set window title

# Function to update the preview of the preprocessed digit
def update_preview(processed_array):
    proc_img = (processed_array.reshape(28, 28) * 255).astype(np.uint8)  # Convert normalized array back to grayscale image
    preview = Image.fromarray(proc_img).resize((100, 100), Image.NEAREST)  # Scale up for visibility
    photo = ImageTk.PhotoImage(preview)  # Create PhotoImage for Tkinter
    preview_label.config(image=photo)  # Update label with new image
    preview_label.image = photo  # Store reference to prevent garbage collection

# Function to display a bar chart of prediction probabilities
def show_prob_chart(probs):
    plt.figure(figsize=(4, 2))  # Create new figure
    plt.bar(range(10), probs)  # Plot bars for each digit class
    plt.xticks(range(10))  # Set x-axis ticks
    plt.title('Prediction Probabilities')  # Add title
    plt.show()  # Show plot interactively

# Function to display the confusion matrix using validation data
def show_confusion():
    y_pred = clf.predict(X_val)  # Predict labels on validation set
    cm = confusion_matrix(y_val, y_pred)  # Compute confusion matrix
    disp = ConfusionMatrixDisplay(cm)  # Prepare display
    disp.plot(cmap='Blues')  # Plot with blue colormap
    plt.show()  # Show plot

# Define accepted file types for upload dialogs
FILETYPES = [
    ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")),  # Common image extensions
    ("All files", ("*.*",))  # All files option
]

# Function to prompt user to select an image file
def ask_image_file():
    tmp = tk.Tk()  # Create temporary Tk window
    tmp.withdraw()  # Hide it immediately
    path = filedialog.askopenfilename(parent=tmp, filetypes=FILETYPES)  # Open file dialog
    tmp.destroy()  # Destroy temporary window
    return path  # Return selected file path

# Function to handle uploaded image and perform prediction
def on_upload(path=None):
    if path is None:  # If no path provided
        path = ask_image_file()  # Prompt user to select file
    if not path:  # If user cancels dialog
        return  # Exit function
    pil_img = Image.open(path).convert('L').resize((100, 100), Image.LANCZOS)  # Load and prepare image for display
    photo = ImageTk.PhotoImage(pil_img)  # Convert to Tkinter-compatible image
    image_label.config(image=photo)  # Show original image
    image_label.image = photo  # Store reference to prevent GC

    data = preprocess_opencv(path)  # Preprocess image data for classifier
    update_preview(data)  # Show processed preview

    pred = clf.predict(data)[0]  # Predict digit class
    conf = np.max(clf.predict_proba(data)) * 100  # Determine confidence percentage
    result_label.config(text=f"Predicted: {pred}\nConfidence: {conf:.1f}%")  # Display prediction and confidence

    show_prob_chart(clf.predict_proba(data)[0])  # Show probability bar chart

# Function to launch a drawing canvas window for manual digit entry
def launch_draw_window():
    draw_win = tk.Toplevel(root)  # Create a new top-level window
    draw_win.title("Draw Your Digit")  # Set window title
    W = H = 200  # Define canvas dimensions

    canvas = tk.Canvas(draw_win, width=W, height=H, bg='white')  # Create drawing canvas
    canvas.pack()  # Pack canvas into window
    pil_img = Image.new('L', (W, H), 'white')  # Create blank PIL image for drawing
    draw   = ImageDraw.Draw(pil_img)  # Create draw object for PIL image
    strokes = []  # List to record strokes for undo
    last    = {'x': None, 'y': None}  # Store last cursor position

    thickness_var = tk.IntVar(value=8)  # Control for brush thickness
    Scale(draw_win, from_=1, to=20, orient='horizontal',  # Create brush size slider
          label='Brush Size', variable=thickness_var).pack()

    # Callback when starting a stroke
    def start_draw(e): last['x'], last['y'] = e.x, e.y  # Record initial point

    # Callback for drawing motion events
    def draw_motion(e):
        x, y = e.x, e.y  # Current cursor position
        w    = thickness_var.get()  # Current brush width
        if last['x'] is not None:  # If there's a previous point
            canvas.create_line(last['x'], last['y'], x, y,  # Draw line on canvas
                               width=w, fill='black',
                               capstyle='round', smooth=True)
            draw.line([last['x'], last['y'], x, y], fill=0, width=w)  # Draw line on PIL image
            strokes.append({'coords':(last['x'], last['y'], x, y), 'w': w})  # Save stroke data
        last['x'], last['y'] = x, y  # Update last position

    # Clear the drawing canvas and stroke history
    def clear_canvas():
        canvas.delete('all')  # Erase all canvas content
        draw.rectangle([0, 0, W, H], fill='white')  # Reset PIL image to blank
        strokes.clear()  # Clear stroke records

    # Undo the last stroke drawn
    def undo():
        if strokes:  # If stroke history exists
            strokes.pop()  # Remove last stroke entry
            canvas.delete('all')  # Clear canvas
            draw.rectangle([0, 0, W, H], fill='white')  # Reset PIL image
            for s in strokes:  # Replay remaining strokes
                x0, y0, x1, y1 = s['coords']  # Stroke coordinates
                w             = s['w']  # Stroke width
                canvas.create_line(x0, y0, x1, y1,
                                   width=w, fill='black',
                                   capstyle='round', smooth=True)  # Redraw on canvas
                draw.line([x0, y0, x1, y1], fill=0, width=w)  # Redraw on PIL image

    # Finish drawing and perform prediction on the drawn digit
    def finish_draw():
        preview = pil_img.resize((100, 100), Image.LANCZOS)  # Resize drawn image for display
        photo   = ImageTk.PhotoImage(preview)  # Convert to Tk image
        image_label.config(image=photo)  # Show in main window
        image_label.image = photo  # Store reference

        gray_np = np.array(pil_img)  # Convert PIL image to NumPy array
        data    = preprocess_opencv(gray_np)  # Preprocess for classifier
        update_preview(data)  # Update processed preview

        pred = clf.predict(data)[0]  # Predict drawn digit
        conf = np.max(clf.predict_proba(data)) * 100  # Calculate confidence
        result_label.config(text=f"Predicted: {pred}\nConfidence: {conf:.1f}%")  # Display result

        show_prob_chart(clf.predict_proba(data)[0])  # Show probabilities
        draw_win.destroy()  # Close drawing window

    # Bind mouse events to drawing callbacks
    canvas.bind('<Button-1>', start_draw)  # Mouse press initiates stroke
    canvas.bind('<B1-Motion>', draw_motion)  # Mouse drag draws line

    # Create control buttons for drawing window
    btn_frame = tk.Frame(draw_win)  # Frame for buttons
    btn_frame.pack(pady=5)  # Pack with padding
    Button(btn_frame, text="Clear", command=clear_canvas).grid(row=0, column=0, padx=5)  # Clear button
    Button(btn_frame, text="Undo",  command=undo).grid(row=0, column=1, padx=5)  # Undo button
    Button(btn_frame, text="Done",  command=finish_draw).grid(row=0, column=2, padx=5)  # Done button

# Main window controls: upload, draw, and view confusion matrix
btn_frame = tk.Frame(root)  # Frame for main buttons
btn_frame.pack(pady=10)  # Pack with vertical padding
Button(btn_frame, text="Upload Digit Image", command=on_upload).grid(row=0, column=0, padx=5)  # Upload button
Button(btn_frame, text="Draw Digit",         command=launch_draw_window).grid(row=0, column=1, padx=5)  # Draw button
Button(btn_frame, text="View Confusion Matrix", command=show_confusion).grid(row=0, column=2, padx=5)  # Confusion matrix button

# Preview label for processed digit image
preview_label = Label(root, text="Preprocessed Preview")  # Label description
preview_label.pack()  # Pack label into window
# Label for displaying the original or drawn image
image_label   = Label(root)  # Empty label for image
image_label.pack(pady=5)  # Pack with padding
# Label for displaying prediction results
result_label  = Label(root, text="No prediction yet", font=("Arial", 14))  # Initial text
result_label.pack(pady=5)  # Pack with padding

# Start the GUI event loop
root.mainloop()  # Begin Tkinter main loop
