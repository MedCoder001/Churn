import pandas as pd
from flask import Flask, request, render_template
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Create a MinMaxScaler
scaler = MinMaxScaler()

@app.route("/", methods=['GET', 'POST'])
def load_page():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'csvFile' not in request.files:
            return render_template('home.html', csvContent="Please select a CSV file.")
    
        file = request.files['csvFile']
    
        # Check if the file is selected
        if file.filename == '':
            return render_template('home.html', csvContent="Please select a CSV file.")
    
        # Read the CSV file content
        csv_content = file.read().decode("utf-8")

        # Read the uploaded CSV data into a DataFrame
        input_df = pd.read_csv(io.StringIO(csv_content))
        
        # Identify columns to scale dynamically
        cols_to_scale = [col for col in input_df.columns if col in ['tenure', 'MonthlyCharges', 'TotalCharges']]
        
        # Replace empty strings with NaN and convert columns to numeric
        input_df[cols_to_scale] = input_df[cols_to_scale].replace('', pd.NA).apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        input_df = input_df.dropna(subset=cols_to_scale)
        
        # Convert 'TotalCharges' to integer (assuming it contains only numeric values)
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(int)

        # Filter numeric columns
        numeric_cols = input_df.select_dtypes(include=['float64', 'int64']).columns

        # Convert numeric columns to float
        input_df[numeric_cols] = input_df[numeric_cols].astype('float32')


        # Scale selected columns
        input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])
        
        # Make predictions
        predictions = model.predict(input_df)

        # Display CSV content and predictions on the webpage
        return render_template('home.html', csvContent=csv_content, predictions=predictions)
    
    return render_template('home.html', csvContent="Upload a CSV file.")

if __name__ == "__main__":
    app.run(debug=True)
