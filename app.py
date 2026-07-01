# File: EEG_Classifier_App/app.py
import os
import numpy as np
import joblib
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from scipy.signal import welch
# import pywt # Only needed if using wavelet features; Welch is used here.

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Relative path to the uploads folder
ALLOWED_EXTENSIONS = {'txt', 'TXT'}
MODEL_FILENAME = "eeg_classifier_model.joblib" # Relative path to the model file
SCALER_FILENAME = "eeg_scaler.joblib" # Relative path to the scaler file
FS_VALUE = 173.61 # IMPORTANT: Must match the value used in train_save_model.py
CLASSES = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E'] # Class names

# --- Initialize Flask App ---
app = Flask(__name__) # Flask finds the 'templates' folder automatically relative to app.py
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Necessary for flashing messages

# --- Create upload folder if it doesn't exist ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created folder: {UPLOAD_FOLDER}")

# --- Load Model and Scaler ---
# Ensure these files are in the same directory as app.py or provide full paths
try:
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    scaler_path = os.path.join(os.path.dirname(__file__), SCALER_FILENAME)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model ({MODEL_FILENAME}) and Scaler ({SCALER_FILENAME}) loaded successfully.")
    # Print model input feature count if available
    if hasattr(model, 'n_features_in_'):
        print(f"Model expects {model.n_features_in_} features.")
except FileNotFoundError:
    print(f"ERROR: Model or Scaler file not found. Expected '{MODEL_FILENAME}' and '{SCALER_FILENAME}' in the same directory as app.py.")
    print("Please run the 'train_save_model.py' script first to generate these files.")
    exit() # Stop the app if model/scaler can't be loaded
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Feature extraction function (must match the one used in training)
def extract_features(data):
    features = []
    # Assuming data is already shaped correctly (e.g., (1, n_timepoints) for prediction)
    for i in range(data.shape[0]):
        # fs is the sampling frequency - MUST BE THE SAME AS TRAINING
        # Use consistent nperseg if possible, or ensure signal length is consistent
        freqs, psd = welch(data[i], fs=FS_VALUE, nperseg=min(256, data.shape[1]))
        features.append(psd)
    features_np = np.array(features)
    print(f"Extracted features shape in prediction: {features_np.shape}") # Debug: Check feature shape
    return features_np

# Prediction function - MODIFIED TO HANDLE NON-NUMERIC DATA
def predict_class(file_path):
    try:
        # --- Modified data loading to handle non-numeric data ---
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Attempt to convert the line to a float
                    # strip() removes leading/trailing whitespace, including potential BOM characters or newlines
                    data.append(float(line.strip()))
                except ValueError:
                    # If conversion fails, skip this line (it's not a valid number)
                    print(f"Skipping non-numeric line: {line.strip()}")
                    pass # Ignore lines that cannot be converted to float

        if not data:
            # No valid numerical data was found in the file
            raise ValueError("Uploaded file contains no valid numerical data or is empty.")

        data = np.array(data)
        print(f"Loaded data shape from file (after cleaning): {data.shape}") # Debug: Check loaded shape
        # --- End of modified data loading ---


        # Reshape for scaler (needs 2D: (n_samples, n_features))
        # Assuming each file contains one signal flattened into a row or a single column
        if data.ndim == 1:
             # If it's a 1D array (e.g., (4097,)), reshape to (1, 4097)
             data_reshaped = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] == 1:
             # Already (1, 4097)
             data_reshaped = data
        elif data.ndim == 2 and data.shape[1] == 1:
             # If it's a column vector (e.g., (4097, 1)), transpose and reshape
             data_reshaped = data.T.reshape(1, -1)
        else:
             # Handle unexpected shapes
             raise ValueError(f"Unexpected data shape loaded: {data.shape}. Expected a 1D array or a single row/column 2D array.")
        print(f"Reshaped data for scaler: {data_reshaped.shape}") # Debug: Check shape for scaler

        # Check if reshaped data has the correct number of time points (features for scaler)
        # scaler.n_features_in_ was determined by the training data (e.g., 4097)
        if hasattr(scaler, 'n_features_in_') and data_reshaped.shape[1] != scaler.n_features_in_:
             raise ValueError(f"Data length mismatch. Scaler expects {scaler.n_features_in_} time points, but file has {data_reshaped.shape[1]}.")

        # Normalize the data using the loaded scaler
        data_normalized = scaler.transform(data_reshaped)
        print(f"Normalized data shape: {data_normalized.shape}") # Debug: Check normalized shape

        # Extract features (using the same method as training)
        features = extract_features(data_normalized) # Expects (n_samples, n_timepoints), returns (n_samples, n_psd_points)

        # Ensure features have the correct shape for the model
        # model.n_features_in_ was determined by the training features (e.g., 129 from Welch)
        if hasattr(model, 'n_features_in_'):
            expected_n_features = model.n_features_in_
            if features.shape[1] != expected_n_features:
                raise ValueError(f"Feature size mismatch for model. Model expects {expected_n_features} features (PSD points), but got {features.shape[1]}. Ensure signal length and Welch parameters match training.")
        # else: # Fallback if attribute doesn't exist (less reliable)
             # print(f"Warning: Model feature count attribute not found. Cannot verify feature input size ({features.shape[1]}).")
             # pass # Allow prediction attempt, may fail if size is wrong.


        # Predict class using the loaded model
        prediction_index = model.predict(features)[0] # Get the first prediction for the single sample
        prediction_proba = None
        if hasattr(model, "predict_proba"):
             prediction_proba = model.predict_proba(features)[0]
             print(f"Probabilities: {prediction_proba}") # Debug: Show probabilities

        # Return the predicted class name
        predicted_class = CLASSES[int(prediction_index)]
        print(f"Prediction: Index={prediction_index}, Class={predicted_class}")
        return predicted_class, prediction_proba # Return both

    except ValueError as ve:
         # Specific value errors from checks above
         print(f"ValueError during prediction: {ve}")
         flash(str(ve), 'error') # Show specific error to user
         return None, None # Indicate error
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        flash("Error: Uploaded file not found during processing.", 'error')
        return None, None
    except Exception as e:
        # Catch other potential errors (e.g., loading non-numeric data, unexpected scaler/model issues)
        print(f"An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        flash(f"Error processing file. Check if it's a valid EEG signal text file. Details: {e}", 'error')
        return None, None


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Render the main page with the upload form
    return render_template('index.html') # No context needed on initial load

@app.route('/predict', methods=['POST'])
def upload_file_and_predict():
    prediction_display = None
    filename_display = None
    probabilities_display = None

    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('index')) # Redirect back to index

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('index')) # Redirect back to index

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # Good practice for security
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            # Use a temporary name to avoid conflicts and potential read issues during save
            temp_filepath = filepath + '.tmp'
            file.save(temp_filepath)
            print(f"File saved temporarily to: {temp_filepath}")

            # Now process the saved temporary file
            # Rename to the final path after successful save if needed, or just use temp_filepath
            # For simplicity, we'll just use the temp_filepath for prediction

            # Make prediction
            predicted_class, prediction_proba = predict_class(temp_filepath)

            if predicted_class is not None:
                filename_display = filename # Display original filename to user
                prediction_display = predicted_class
                if prediction_proba is not None:
                    probabilities_display = {CLASSES[i]: f"{prob*100:.2f}%" for i, prob in enumerate(prediction_proba)}

            # Clean up the uploaded temporary file after prediction
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                print(f"Temporary file removed: {temp_filepath}")


        except Exception as e:
            # Catch potential errors during file save or prediction call
            print(f"Error during file save or prediction call: {e}")
            flash(f'Error processing file: {e}', 'error')
            # Clean up potentially corrupt temporary file if it exists
            if os.path.exists(temp_filepath):
                 try:
                     os.remove(temp_filepath)
                     print(f"Cleaned up potentially corrupt temporary file: {temp_filepath}")
                 except OSError as oe:
                     print(f"Error removing file {temp_filepath}: {oe}")

        # Render the page again, passing prediction results (or None if error)
        return render_template('index.html',
                               prediction=prediction_display,
                               filename=filename_display,
                               probabilities=probabilities_display)

    else:
        flash('Invalid file type. Allowed types are: .txt, .TXT', 'error')
        return redirect(url_for('index')) # Redirect back to index if file type is wrong

# --- Run the App ---
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on your network, default is 127.0.0.1 (localhost only)
    app.run(debug=True, host='0.0.0.0', port=5001)
 # debug=True for development, set to False for production