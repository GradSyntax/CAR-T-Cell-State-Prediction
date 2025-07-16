import pandas as pd
from scipy.sparse import load_npz
import joblib
from comet_ml import Experiment # Changed from start for clarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. Your Personal Comet Experiment ---
experiment = Experiment(
    api_key="VOLKCzWGvfRFofscOQFpPMW09",
    project_name="car-t-cell-classifier",
    workspace="grad-syntax"
)

# --- 2. Load Data ---
print("Loading data...")
X = load_npz("./data/X_data.npz")
y = pd.read_csv("./data/y_labels.csv").values.ravel()
print("Data loaded.")

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Data split complete.")

# --- 4. Train Model ---
params = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 100,
    "random_state": 42
}
experiment.log_parameters(params) # Log all params to Comet

print("Training model...")
# Create the model using only the parameters it understands
model = RandomForestClassifier(
    n_estimators=params['n_estimators'],
    random_state=params['random_state']
)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate Model ---
print("Evaluating model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
experiment.log_metric("accuracy", accuracy)

# --- 6. Save and Log Model ---
model_filename = "cell_classifier_model.joblib"
joblib.dump(model, model_filename)
experiment.log_model("RandomForest", model_filename)
print(f"Model saved as {model_filename}")

experiment.end()

# --- 7. Find and Log Important Genes (Interpretability) ---
print("Finding most important genes...")

# Load the list of 2000 highly variable gene names
gene_names = pd.read_csv("./data/hvg_names.csv", header=None).values.ravel()

# Create a report of gene importances from the trained model
feature_importances = pd.DataFrame(
    model.feature_importances_, 
    index=gene_names, 
    columns=['importance']
).sort_values('importance', ascending=False)

# Save the top 20 most important genes to a local CSV file
feature_importances.head(20).to_csv("top_20_important_genes.csv")

# Log this table directly to your Comet project
experiment.log_table("top_20_important_genes.csv")

print("Top 20 most important genes logged to Comet.")