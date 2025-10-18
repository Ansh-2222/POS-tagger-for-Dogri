from main import load_data, prepare_dataset, train_svm  # replace 'main' with your actual model file

import joblib
import os

data_path = "DogriDataset.txt"
sentences = load_data(data_path)
X, y = prepare_dataset(sentences)

clf, le, vec = train_svm(X, y)

# Save model files
os.makedirs("model_files", exist_ok=True)
joblib.dump(clf, "model_files/model.pkl")
joblib.dump(le, "model_files/label_encoder.pkl")
joblib.dump(vec, "model_files/vectorizer.pkl")

print("✅ Model files saved successfully in model_files/")
