import gradio as gr
from PIL import Image,ImageOps
"""def crop_center(image, cropx, cropy):
    y, x, _ = image.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return image[starty:starty + cropy, startx:startx + cropx]"""


def predict_digit(frame):
    try:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return "No face detected."

        x, y, w, h = faces[0]  # First face only
        face = frame[y:y+h, x:x+w]  # Crop face
        # Resize the image using PIL before converting to NumPy array

        #frame = crop_center(frame, 20, 20)
        image = Image.fromarray(face.astype('uint8'), 'RGB').resize((128, 128))
        image = np.array(image) / 255.0
        image = image.reshape(1, 128, 128, 3)

        prediction = model.predict(image)
        #pred = prediction[0]

        label = "With Mask 😷" if prediction < 0.5 else "Without Mask 😐"
        confidence = (1 - prediction) if prediction < 0.5 else prediction

        return f"{label}\nConfidence: {np.max(prediction)*100:0.1f}"
    except Exception as e:
      return f"Error: {e}"
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(),
    outputs="text",
    title="Digit Classifier",
    description="Draw a digit (0–9) and let the model predict it!"
)
demo.launch(share=True)