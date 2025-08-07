# **Face Mask Detection**

This project is a deep learning application that detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN). The system uses image classification techniques to determine if a face is with or without a mask.

---

## 📊 Dataset

- **Source**: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data
  - With Mask  
  - Without Mask  
- 


---

## 🔁 Project Workflow

1. **Data Preprocessing**  
2. **Model Building (CNN using TensorFlow/Keras)**  
3. **Model Training & Evaluation**  
4. **Deployment with Gradio Interface**

---

## 🏆 Results

After training and evaluating the CNN model on the face mask dataset:

- 🥇 **Best Performing Model**: `CNN (with Conv2D, MaxPooling, Dropout layers)`
- ✅ **Achieved Accuracy**: **Over 95% on validation data**
- 🧠 Techniques used:
- Image Augmentation with `ImageDataGenerator`
- Optimization with `Adam`
- Loss Function: `CategoricalCrossentropy`

---

## Model Gradio view
 

![Image](https://github.com/user-attachments/assets/45bf848a-3664-4aec-b559-707aa58ffb58)




![Image](https://github.com/user-attachments/assets/dbce016c-afaa-4395-b975-c411d20bd5ec)

---
## 🧪 Requirements

To run this project, install the required libraries:

```bash
pip install tensorflow gradio opencv-python pillow
pip install kagglehub 



