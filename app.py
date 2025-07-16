import gradio as gr
import joblib
import pandas as pd
from scipy.sparse import load_npz
import numpy as np

# Load the saved model and a small subset of the data for the demo
model = joblib.load("cell_classifier_model.joblib")
X_data = load_npz("./data/X_data.npz")

def predict_random_cell():
    # Select a random cell from the dataset
    random_index = np.random.randint(0, X_data.shape[0])
    cell_data = X_data[random_index]

    # Make a prediction
    # The model expects a 2D array, so we reshape it
    prediction = model.predict(cell_data.reshape(1, -1))[0]

    return f"Selected a random cell from the dataset.\n\nModel Prediction: This cell belongs to Cluster {prediction}."

# Create the Gradio web interface
iface = gr.Interface(
    fn=predict_random_cell,
    inputs=None, # The user doesn't need to provide input
    outputs=gr.Textbox(label="Prediction Result", lines=3),
    title="CAR-T Cell Cluster Predictor",
    description="Click the 'Submit' button to select a random cell from our dataset and see the model's prediction for which cluster it belongs to."
)

# Launch the app
iface.launch()