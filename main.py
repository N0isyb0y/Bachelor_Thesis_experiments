from flask import Flask, jsonify, request, render_template, Response
import os
import pm4py
from werkzeug.utils import secure_filename
from pattern_miner import *
import pandas as pd
import json
from analysis import *

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store event log data
event_log_data = None
filename = ""

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/data', methods=['GET'])
def get_data():
    # Daten, die das Frontend (z.B. D3.js) verwendet
    data = {
        'values': [10, 20, 30, 40, 50]
    }
    return jsonify(data)

@app.route('/upload', methods=['POST'])
def upload_file():
    global event_log_data
    global filename
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if the uploaded file is XES (optional but recommended)
    if not (file.filename.endswith('.xes') or file.filename.endswith('.xes.gz')):
        return jsonify({'error': 'Invalid file format. Please upload an XES file.'}), 400

    # Secure the filename and save it to the upload folder
    filename_temp = secure_filename(file.filename)
    filename = filename_temp
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_temp)
    file.save(file_path)

    try:
        # Use pm4py to load the XES file from the saved path
        event_log = pm4py.read_xes(file_path)
        event_log_data  = event_log
        # Example: Get basic statistics about the event log
        number_of_events = len(event_log)
        start_activities = pm4py.get_start_activities(event_log)
        end_actvities = pm4py.get_end_activities(event_log)
        print("Start activities: {}\nEnd activities: {}".format(start_activities, end_actvities))

    except Exception as e:
        print(f"Error processing XES file: {e}")  # Log error to Flask console
        return jsonify({'error': f'Failed to parse XES file: {str(e)}'}), 500

    # Optional: Remove the file after processing (clean-up)
    os.remove(file_path)

    # Log some information to the Flask console for debugging
    print(f"Uploaded XES file processed successfully.")
    print(f"Number of events: {number_of_events}, Number of cases: {number_of_events}")


    options_for_events = list(event_log['concept:name'].unique())
    # Return a success message to the frontend
    return jsonify(options_for_events)

@app.route('/process_selections', methods=['POST'])
def process_selections():
    global event_log_data
    global filename

    if event_log_data is None:
        return jsonify({"error": "No event log data found. Please upload a file first."}), 400
    
    data = request.json
    """Receives and handles both dropdown selections."""
    selection1 = data.get('selection1')
    selection2 = data.get('selection2')
    analysis_option = data.get('analysis')
    gamma = data.get('gamma')
    batch_type = data.get('batchType')
    min_Pattern_Size = data.get('minPatternSize')
    min_Sample = data.get('minSample')
    violation_Tolerance = data.get('violationTolerance')
    time_unit = data.get('timeUnit')  # Get time unit from frontend

    #data = prepare_data(event_log_data, selection1, selection2)
    #print(data)
    time_conversion = {
        "seconds": 1 / 86400,  # 1 second = 1/86400 days
        "minutes": 1 / 1440,   # 1 minute = 1/1440 days
        "hours": 1 / 24,       # 1 hour = 1/24 days
        "days": 1              # 1 day = 1 day
    }

    gamma_days = gamma * time_conversion.get(time_unit, 1)  # Default to days if unknown unit


    print(f"Dropdown1 selection: {selection1}")
    print(f"Dropdown2 selection: {selection2}")
    print(f"Analysis option: {analysis_option}")
    print(f"Batch type: {batch_type}")
    print(f"Gamma value: {gamma}")
    filename2 = filename + "_" + selection1 + "_" + selection2 + "_" + analysis_option + "_" + batch_type + "_" + str(gamma)

    try:
        # Determine the processing function based on user selections
        if analysis_option == 'Duration':
            data_processed, data_unprocessed = sequence_DBSCAN(selection1, selection2, event_log_data , gamma_days, min_Pattern_Size)
        elif batch_type == 'lifo':
            data_processed, data_unprocessed = Batch_on_end_lifo_old(selection1, selection2, event_log_data, min_Pattern_Size)
        elif analysis_option == 'Batch on end v1':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = Batch_on_end_fifo2(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size, violation_Tolerance)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = Batch_on_end_unordered(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size)        
        elif analysis_option == 'Batch on end v2':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = new_Batch_on_end_fifo_3(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = new_Batch_on_end_unordered(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size)
        elif analysis_option == 'Batch on end DBSCAN':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = Batch_on_end_fifo_DBSCAN2(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size, min_Sample)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = Batch_on_end_unordered_DBSCAN(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size, min_Sample)
        elif analysis_option == 'Batch on start v1':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = Batch_on_start_fifo(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = Batch_on_start_unordered(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size)
        elif analysis_option == 'Batch on start v2':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = new_Batch_on_start_fifo(selection1, selection2, event_log_data, min_Pattern_Size, gamma_days)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = new_Batch_on_start_unordered(selection1, selection2, event_log_data, min_Pattern_Size, gamma_days)
        elif analysis_option == 'Batch on start DBSCAN':
            if batch_type == 'fifo':
                data_processed, data_unprocessed = Batch_on_start_fifo_DBSCAN(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size, min_Sample)
            elif batch_type == 'unordered':
                data_processed, data_unprocessed = Batch_on_start_unordered_DBSCAN(selection1, selection2, event_log_data, gamma_days, min_Pattern_Size, min_Sample)
        
        
        concat_df = pd.concat(data_processed, ignore_index=True)
        #total_sequence = prepare_data(event_log_data, selection1, selection2, "end")
        # Assuming `prepared_df` is your first DataFrame and `subset_df` is the subset DataFrame

        # Create a new column in prepared_df and set all rows to 0
        data_unprocessed['is_in_subset'] = 0

        # Update the column to 1 for cases that are in the subset_df
        data_unprocessed.loc[data_unprocessed['case_ID'].isin(concat_df['case_ID']), 'is_in_subset'] = 1


        total_sequence_json = data_unprocessed.to_json(orient="records")

        print(data_unprocessed)

        
        if isinstance(data_processed, list) and all(isinstance(df, pd.DataFrame) for df in data_processed):
            # Concatenate the DataFrames in the lis
            
            # Convert the concatenated DataFrame to JSON
            result_json = concat_df.to_json(orient='records')
            
            metrics = {}
            if isinstance(data_processed, list) and all(isinstance(df, pd.DataFrame) for df in data_processed):
                metrics["number_of_traces"] = number_of_traces(concat_df)
                metrics["number_of_batches"] = number_of_batches(data_processed)
                metrics["batch_frequency"] = batch_frequency(data_unprocessed, data_processed)
                metrics["batch_duration_max"] = batch_duration_max(data_processed)
                metrics["average_batch_arrival_window"] = average_batch_arrival_window(data_processed)
                metrics["average_batch_interval"] = average_batch_interval(data_processed)
            
           
            return jsonify({"status": "success", "data": json.loads(result_json), "total_sequence": json.loads(total_sequence_json), "analysis_option": analysis_option, "metrics": metrics}), 200
        else:
            return jsonify({"error": "Processed data is not a valid list of DataFrames."}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/json')
def get_json_data():
    
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'selections.json')) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
