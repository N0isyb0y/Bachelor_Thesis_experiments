import pandas as pd
def number_of_traces(event_log):
    return len(event_log)

def number_of_batches(batches):
    return len(batches)

def batch_frequency(original_event_log, batches):
    combined_df = pd.concat(batches, ignore_index=True)
    return (len(combined_df) / number_of_traces(original_event_log))*100

def batch_duration_max(batches):
    max_duration = pd.Timedelta(0)  # Start with a zero Timedelta
    for batch in batches:
        batch_max_duration = batch['duration'].max()
        if batch_max_duration > max_duration:
            max_duration = batch_max_duration
    
    # Convert max_duration to float representing days
    max_duration_in_days = max_duration.total_seconds() / (24 * 60 * 60)  # Convert seconds to days
    return max_duration_in_days

def average_batch_arrival_window(batches):
    total_window = 0
    num_batches = len(batches)

    for batch in batches:
        # Calculate max and min end_time in the batch
        max_end_time = batch['end_time'].max()
        min_end_time = batch['end_time'].min()
        
        # Calculate the window for this batch
        batch_window = (max_end_time - min_end_time).total_seconds()  # Get window in seconds
        total_window += batch_window
        
    average_window = (total_window / num_batches) / (24 * 3600)
    return average_window

def average_batch_interval(batches):
    
    total_interval = 0
    num_intervals = len(batches) - 1  # Number of intervals is one less than number of batches

    if len(batches) == 1:
        return 0
    
    for i in range(num_intervals):
        # Get the current batch and the next batch
        current_batch = batches[i]
        next_batch = batches[i + 1]
        
        # Find last trace time of the current batch and first trace time of the next batch
        last_trace_current_batch = current_batch['end_time'].max()
        first_trace_next_batch = next_batch['end_time'].min()
        
        # Calculate the interval
        batch_interval = (first_trace_next_batch - last_trace_current_batch).total_seconds()  # Seconds
        total_interval += batch_interval
    
    # Calculate the average interval (convert seconds to days for readability)
    average_interval = (total_interval / num_intervals) / (24 * 3600)
    return average_interval

def calculate_average_batch_size(batches):
    total_traces = sum(len(batch) for batch in batches)
    num_batches = len(batches)
    return total_traces / num_batches