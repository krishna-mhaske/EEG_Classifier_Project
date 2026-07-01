# File: EEG_Classifier_App/train_save_model.py

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from scipy.signal import welch
import pywt
import matplotlib.pyplot as plt
import joblib # Needed for saving

# %%
# Function to load data from a folder
def load_data_from_folder(folder_path, file_prefix, file_extension='txt'):
    data = []
    print(f"Attempting to load from: {folder_path}") # Debug print
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found - {folder_path}")
        return np.array(data) # Return empty array if folder doesn't exist

    files_loaded_count = 0
    for i in range(1, 101):
        filename = f"{file_prefix}{i:03d}.{file_extension}"
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                data.append(np.loadtxt(file_path))
                files_loaded_count += 1
            except Exception as e:
                 print(f"Error loading file {file_path}: {e}")
        # else:
            # print(f"File not found: {file_path}") # Optional: uncomment for verbose missing file logs
    print(f"Loaded {files_loaded_count} files from {folder_path}")
    return np.array(data)

# %%
# Updated paths based on your provided location:
base_path = r'C:\Users\91915\OneDrive\Desktop\TY\sem2\CC\cp\EEG_Prediction_Project\dataset'

A_folder_path = rf'{base_path}\SET A\Z' # <- Should point to the Z folder inside SET A
B_folder_path = rf'{base_path}\SET B\O' # <- Should point to the O folder inside SET B
C_folder_path = rf'{base_path}\SET C\N' # <- Should point to the N folder inside SET C
D_folder_path = rf'{base_path}\SET D\F' # <- Should point to the F folder inside SET D
E_folder_path = rf'{base_path}\SET E\S' # <- Should point to the S folder inside SET E
# =====================================================
# Load the datasets from folders
data_A = load_data_from_folder(A_folder_path, 'Z')
data_B = load_data_from_folder(B_folder_path, 'O')
data_C = load_data_from_folder(C_folder_path, 'N', 'TXT')  # Handling '.TXT' extension for Set C
data_D = load_data_from_folder(D_folder_path, 'F')
data_E = load_data_from_folder(E_folder_path, 'S')

# Print the shapes of the loaded datasets for debugging
print(f'data_A shape: {data_A.shape}')
print(f'data_B shape: {data_B.shape}')
print(f'data_C shape: {data_C.shape}')
print(f'data_D shape: {data_D.shape}')
print(f'data_E shape: {data_E.shape}')

# Check if data is loaded properly
if data_A.size == 0 or data_B.size == 0 or data_C.size == 0 or data_D.size == 0 or data_E.size == 0:
    raise ValueError("Data loading error: One or more datasets are empty. Please check the folder paths and ensure data files exist.")

# %%
# Assign labels to the data
labels_A = np.zeros(data_A.shape[0])
labels_B = np.ones(data_B.shape[0])
labels_C = np.full(data_C.shape[0], 2)
labels_D = np.full(data_D.shape[0], 3)
labels_E = np.full(data_E.shape[0], 4)

# Combine data and labels
# Use concatenate for potentially different lengths (although they are all 4097 in this dataset)
# Ensure they are treated as lists of arrays before vstack if shapes differ in columns
data = np.vstack((data_A, data_B, data_C, data_D, data_E))
labels = np.concatenate((labels_A, labels_B, labels_C, labels_D, labels_E))

# Ensure the data is in 2D shape for normalization (should already be if loaded correctly)
# If loaded files have different lengths, this vstack/reshape might fail earlier.
# Assuming all files have 4097 data points as per the dataset description.
print(f"Combined data shape before reshape: {data.shape}") # Shape should be (500, 4097)
if data.shape[1] != 4097:
     print(f"Warning: Expected 4097 columns per sample, but got {data.shape[1]}. Check file loading.")

# Standard scaler works on features (columns), expects (n_samples, n_features)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
print(f"Data normalized shape: {data_normalized.shape}")

# %%
# Extract features using power spectral density (PSD)
FS_VALUE = 173.61 # Define Sampling frequency (important!)

def extract_features(data):
    features = []
    for i in range(data.shape[0]):
        # fs is the sampling frequency
        freqs, psd = welch(data[i], fs=FS_VALUE, nperseg=min(256, data.shape[1])) # Added nperseg for safety
        features.append(psd)
    # Check if PSD lengths are consistent
    psd_lengths = [len(p) for p in features]
    if len(set(psd_lengths)) > 1:
        print(f"Warning: Inconsistent PSD lengths found: {set(psd_lengths)}. This might cause issues.")
        # Implement padding/truncation here if needed, or adjust welch parameters.
        # For now, let's assume they are consistent based on nperseg.
    return np.array(features)

features = extract_features(data_normalized)
print(f"Extracted features shape: {features.shape}") # Should be (500, num_psd_points) e.g. (500, 129)

# Check for NaN or Inf values in features
if np.isnan(features).any() or np.isinf(features).any():
    print("Warning: NaN or Inf values found in features after Welch calculation. Check input data and Welch parameters.")
    features = np.nan_to_num(features) # Replace NaN with 0 and Inf with large finite numbers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# %%
# Define classifiers with hyperparameter grids
classifiers = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
            # Reduced grid for faster training demonstration
        }
    },
    "Support Vector Machine": {
        "model": SVC(probability=True), # Enable probability for potential future use
        "params": {
            "C": [1, 10],
            "gamma": [0.1, 0.01],
            "kernel": ['rbf'] # RBF is often best for this type of data
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {} # GaussianNB usually doesn't need extensive tuning
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }
}

# %%
# Train and evaluate each classifier with Grid Search
best_model = None
best_accuracy = 0
best_name = ""

for name, clf_info in classifiers.items():
    print(f"--- Training {name} ---")
    # Handle cases where params might be empty (like Naive Bayes)
    if clf_info["params"]:
        grid_search = GridSearchCV(clf_info["model"], clf_info["params"], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        current_best_estimator = grid_search.best_estimator_
    else:
        # If no params to search, just fit the model directly
        current_best_estimator = clf_info["model"]
        current_best_estimator.fit(X_train, y_train)

    y_pred = current_best_estimator.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    if accuracy > best_accuracy:
        print(f"New best model found: {name} with accuracy {accuracy:.4f}")
        best_accuracy = accuracy
        best_model = current_best_estimator
        best_name = name

print(f'\nBest Classifier Found: {best_name} with Accuracy: {best_accuracy:.4f}')

# Final evaluation report of the overall best model
if best_model is not None:
    y_pred_best = best_model.predict(X_test)
    print("\n--- Classification Report for Best Model ---")
    print(classification_report(y_test, y_pred_best, target_names=['Set A', 'Set B', 'Set C', 'Set D', 'Set E'], zero_division=0))
else:
    print("\nNo model was successfully trained.")


# %%
# -------- SAVE THE BEST MODEL AND THE SCALER --------
MODEL_FILENAME = "eeg_classifier_model.joblib"
SCALER_FILENAME = "eeg_scaler.joblib"

if best_model is not None:
    # Save the best model
    print(f"\nSaving the best model ({best_name}) to {MODEL_FILENAME}")
    joblib.dump(best_model, MODEL_FILENAME)

    # Save the scaler
    print(f"Saving the scaler to {SCALER_FILENAME}")
    joblib.dump(scaler, SCALER_FILENAME)

    print("\nModel and scaler saved successfully in the current directory.")
else:
    print("\nCannot save model/scaler because no model was determined as best.")

# %% Optional: Visualizations (can be commented out if not needed for saving)
# plt.figure(figsize=(15, 10))
# class_names = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
# datasets = [data_A, data_B, data_C, data_D, data_E]
# for i, dataset in enumerate(datasets):
#     if dataset.size > 0:
#         plt.subplot(3, 2, i + 1)
#         plt.plot(dataset[0]) # Plot first sample from each class
#         plt.title(f'Sample Signal from {class_names[i]}')
# plt.tight_layout()
# plt.show()

# Plot feature importance if Random Forest is the best model
# Note: SVM (with non-linear kernel) and Naive Bayes don't have direct feature_importances_ attribute like trees.
# if best_name == "Random Forest" and hasattr(best_model, 'feature_importances_'):
#     importances = best_model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     plt.figure(figsize=(15, 5))
#     plt.title(f"Feature Importances ({best_name})")
#     plt.bar(range(X_train.shape[1]), importances[indices])
#     plt.xlabel("Feature Index (Sorted by Importance)")
#     plt.ylabel("Importance")
#     plt.xticks(range(X_train.shape[1]), indices) # Show original indices
#     plt.xlim([-1, min(50, X_train.shape[1])]) # Show top 50 features
#     plt.show()
# elif best_name == "Decision Tree" and hasattr(best_model, 'feature_importances_'):
#      importances = best_model.feature_importances_
#      indices = np.argsort(importances)[::-1]
#      plt.figure(figsize=(15, 5))
#      plt.title(f"Feature Importances ({best_name})")
#      plt.bar(range(X_train.shape[1]), importances[indices])
#      plt.xlabel("Feature Index (Sorted by Importance)")
#      plt.ylabel("Importance")
#      plt.xticks(range(X_train.shape[1]), indices)
#      plt.xlim([-1, min(50, X_train.shape[1])])
#      plt.show()

print("\nTraining script finished.")