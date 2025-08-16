# Transport Image Classification Web App

This project is a web application that classifies images into three categories: **Boat**, **Airplane**, and **Car**. The application uses a deep learning model built with **FastAI** and provides real-time predictions via a **Streamlit** interface.

## Demo

- Live Web App: https://transportclassification-ckewhztyobxmvsjtjjkkam.streamlit.app/  
- GitHub Repository: https://github.com/ilhomPython571/Transport_Classification

## Features

- Upload an image (PNG, JPEG, GIF, SVG) to classify.
- Predicts whether the image is a Boat, Airplane, or Car.
- Displays the probability for each class.
- Interactive bar chart visualization of class probabilities using **Plotly**.

## Dataset

The model was trained using the **OIDv4_ToolKit** dataset downloader, fetching 200 images per class for:

- Car  
- Airplane  
- Boat  

## Technologies & Libraries Used

- Python 3.x  
- FastAI  
- Streamlit  
- Plotly  
- PIL (Python Imaging Library)  
- ipywidgets  

## Model

- CNN model based on **ResNet34** architecture.  
- Trained with FastAI `cnn_learner` and fine-tuned for 4 epochs.  
- Saved as `transport_model.pkl` for inference in the web app.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ilhomPython571/Transport_Classification

2. Install required packages:
 ```bash
   pip install -r requirements.txt

4. Run the Streamlit app:
   streamlit run app.py
5. Upload an image and see the classification result along with class probabilities..



