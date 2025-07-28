# DeepLearning-CNN
**Projects:**

#1. Face Anti-Spoofing Authentication System

A real-time AI-powered facial liveness detection system built with Convolutional Neural Networks (CNNs) to detect spoof attacks using a webcam feed. This system distinguishes between real human faces and spoofed attempts (e.g., photos, videos) and includes an interface to enroll and verify faces.

## Features

- ✅ Detects real vs. spoof (fake) faces in real-time.
- ✅ Built using CNN for liveness detection.
- ✅ Streamlit interface with webcam access.
- ✅ User face enrollment support.
- ✅ Trained on real-world datasets with mixed spoof/real samples.

## Tech Stack

| Component     | Tools/Technologies                        |
|---------------|--------------------------------------------|
| Language      | Python                                     |
| Interface     | Streamlit                                  |
| Model         | CNN (Convolutional Neural Network)         |
| Libraries     | OpenCV, TensorFlow/Keras, NumPy, Pandas    |
| Dataset       | Custom real/fake mixed dataset with depth/color images |

---

## Project Structure

Face-Anti-Spoofing/ ├── app_interface.py       
                    ├── train_model.py  
                    ├── data/ 
                            ├── train/ 
                                   ├── color/ 
                                   └── depth/ 
                            │── test/ 
                                   ├── color/
                                   └── depth/ 
                    ├── enrolled_faces/      


## 📊 Dataset:
This project uses the CASIA-FASD (Face Anti-Spoofing Database) for training and evaluating spoof detection.

🧾 Dataset Details:
Developed by Institute of Automation, Chinese Academy of Sciences
Contains videos of real and spoofed face presentations
Spoof types include printed photos, cut photos, and replayed videos
Collected in three resolutions: low, normal, high
Each sample is labeled as real or fake
Used for face liveness detection and spoof attack classificati

📥 Download Link & Access

> 🚫 Note: The CASIA-FASD dataset is not publicly downloadable via a direct link.
To access it officially:

Visit the dataset request page:
🔗 CASIA-FASD Dataset Request

Send a dataset access request to the maintainers
----

## Getting Started

### Installation

 Training the Model

python train_model.py

Model will be saved as anti_spoof_model.h5 after training.

Running the App

streamlit run app_interface.py

> This opens a webcam-powered GUI where you can test spoof detection and enroll faces.

Requirements

streamlit
opencv-python
tensorflow
numpy
pandas
scikit-learn

Install all dependencies with:

pip install -r requirements.txt

Sample Output

> Real-time detection with bounding box and result: "Real" or "Spoof".

---

Model Highlights

Input image resized to 224x224 for model compatibility.
CNN trained with:

Augmentation
Normalization
Early stopping and validation

Performance: High validation accuracy (>98%) on custom dataset

---
Use Cases

Secure phone/PC login

ATM or kiosk authentication

Access control in security zones

---
License

This project is licensed under the MIT License.
