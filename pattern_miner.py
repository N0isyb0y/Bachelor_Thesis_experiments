import pm4py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from itertools import combinations
#from . import analysis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from analysis import *

def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    start_activities = pm4py.get_start_activities(event_log)
    end_actvities = pm4py.get_end_activities(event_log)
    #print("Start activities: {}\nEnd activities: {}".format(start_activities, end_actvities))
    return event_log

def prepare_data(event_log, start_activity, end_activity,batch_type):
    event_log = event_log.sort_values(by=['case:concept:name', 'time:timestamp'])

    filtered_cases = []
    test = []

    for case_id, case_data in event_log.groupby('case:concept:name'):
        # Step 3: Iterate through the events of each case to find "start_activity" followed by "end_activity"
        case_data = case_data.reset_index(drop=True)  # Reset index to easily iterate through rows
        for i in range(len(case_data) - 1):
            # Check if current event is "Create Fine" and the next event is "Payment"
            if (case_data.loc[i, 'concept:name'] == start_activity and
                case_data.loc[i + 1, 'concept:name'] == end_activity):
                filtered_cases.append(case_id)
                test.append(case_data.loc[i])
                test.append(case_data.loc[i+1])
                break  # No need to check further once the sequence is found
    df = pd.DataFrame(test, columns=['time:timestamp', 'case:concept:name','concept:name'])
    
    # Step 4: Filter the original DataFrame to include only the cases that match the condition
    filtered_event_log = event_log[event_log['case:concept:name'].isin(filtered_cases)]
    #print(filtered_event_log)
    prepared_df = pd.DataFrame(data={
            "case_ID": df.iloc[::2]['case:concept:name'].array,
            "start_time": df.iloc[::2]['time:timestamp'].array,
            "end_time": df.iloc[1::2]['time:timestamp'].array,
            "duration": df['time:timestamp'].diff().iloc[1::2].array,
        })

    if(batch_type == "end"):
        sorted_merged_log = prepared_df.sort_values(by=['end_time', 'start_time'], ascending=[True, True])
        sorted_merged_log = sorted_merged_log.reset_index(drop=True)
        #print(sorted_merged_log)
        #sorted_merged_log.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\test4.csv")
    elif(batch_type == "start"):
        sorted_merged_log = prepared_df.sort_values(by=['start_time', 'end_time'], ascending=[True, True])
        sorted_merged_log = sorted_merged_log.reset_index(drop=True)
    return sorted_merged_log

def sequence_DBSCAN(start_activity, end_activity, event_log, gamma = 0.0001, min_batch = 20):
    gamma = gamma * 86400
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    #sorted_merged_log['timestamp_numeric'] = pd.to_datetime(sorted_merged_log["end_time"]).astype('int64') / 10**9
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['duration'].apply(lambda x: int(x.total_seconds()))



    X = sorted_merged_log[['timestamp_numeric']].values

    #scaler = StandardScaler() #optional
    #X_scaled = scaler.fit_transform(X)
    

    dbscan = DBSCAN(eps=gamma, min_samples= 2, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    #batches = [sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric') for label in cluster_labels]

    batches = []
    for label in cluster_labels:
        cluster_data = sorted_merged_log[sorted_merged_log['cluster'] == label]
        
        # If the cluster size is larger than the minimum size, keep it
        if len(cluster_data) >= min_batch:
            batches.append(cluster_data.drop(columns='timestamp_numeric'))
    return batches, sorted_merged_log

def Batch_on_end_unordered_DBSCAN(start_activity, end_activity, event_log, gamma = 0.0001, min_batch = 20, min_samples = 2):
    gamma = gamma * 86400
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    #sorted_merged_log['timestamp_numeric'] = pd.to_datetime(sorted_merged_log["end_time"]).astype('int64') / 10**9
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp())) #convert to unix seconds


    X = sorted_merged_log[['timestamp_numeric']].values

    #scaler = StandardScaler() #optional
    #X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=gamma, min_samples= min_samples, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    #batches = [sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric') for label in cluster_labels]

    batches = []
    for label in cluster_labels:
        cluster_data = sorted_merged_log[sorted_merged_log['cluster'] == label]
        
        # If the cluster size is larger than the minimum size, keep it
        if len(cluster_data) >= min_batch:
            batches.append(cluster_data.drop(columns='timestamp_numeric'))
    return batches, sorted_merged_log

def Batch_on_start_unordered_DBSCAN(start_activity, end_activity, event_log, gamma = 0.0001, min_batch= 20, min_samples = 2):
    gamma = gamma * 86400
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    #sorted_merged_log['timestamp_numeric'] = pd.to_datetime(sorted_merged_log["end_time"]).astype('int64') / 10**9
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp())) #convert to unix seconds


    X = sorted_merged_log[['timestamp_numeric']].values

    #scaler = StandardScaler() #optional
    #X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=gamma, min_samples= min_samples, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    #batches = [sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric') for label in cluster_labels]

    batches = []
    for label in cluster_labels:
        cluster_data = sorted_merged_log[sorted_merged_log['cluster'] == label]
        
        # If the cluster size is larger than the minimum size, keep it
        if len(cluster_data) >= min_batch:
            batches.append(cluster_data.drop(columns='timestamp_numeric'))
    return batches, sorted_merged_log


def Batch_on_end_fifo_DBSCAN2(start_activity, end_activity, event_log, gamma=0.000001, min_batch=20, min_samples=2):
    batches = []
    
    # Convert gamma from days to seconds
    gamma = gamma * 86400  

    # Filter relevant columns and prepare event log
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp()))

    # Step 1: Run DBSCAN first to form initial clusters
    X = sorted_merged_log[['timestamp_numeric']].values
    dbscan = DBSCAN(eps=gamma, min_samples=min_samples, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    filtered_batches = []

    # Step 2: Iterate over clusters and filter each batch for FIFO
    for label in cluster_labels:
        batch = sorted_merged_log[sorted_merged_log["cluster"] == label].drop(columns="cluster")

        # Find longest increasing subsequence within each batch (FIFO filtering)
        batch = batch.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

        valid_indices = find_largest_increasing_subsequence2(batch["end_time"].values)
        filtered_batch = batch.iloc[valid_indices]

        # If filtered batch is not empty, store it
        if len(filtered_batch) >= min_batch:
            filtered_batches.append(filtered_batch)

    # Step 3: Re-run DBSCAN on each filtered batch to check if traces are still close
    final_batches = []
    for batch in filtered_batches:
        if len(batch) < min_batch:  # Skip small batches
            continue

        # Convert timestamps again for DBSCAN
        batch = batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)


        batch['timestamp_numeric'] = batch['end_time'].apply(lambda x: int(x.timestamp()))

        X = batch[['timestamp_numeric']].values
        dbscan = DBSCAN(eps=gamma, min_samples=min_samples, metric='euclidean')
        batch['cluster'] = dbscan.fit_predict(X)

        for label in batch['cluster'].unique():
            if label == -1:  # Ignore noise
                continue

            final_batch = batch[batch["cluster"] == label].drop(columns=["cluster", "timestamp_numeric"])
            if len(final_batch) >= min_batch:
                final_batches.append(final_batch)

    return final_batches, sorted_merged_log

def Batch_on_start_fifo_DBSCAN(start_activity, end_activity, event_log, gamma=0.000001, min_batch=20, min_samples=20):
    batches = []
    
    # Convert gamma from days to seconds
    gamma = gamma * 86400  

    # Filter relevant columns and prepare event log
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="start")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp()))

    # Step 1: Run DBSCAN first to form initial clusters
    X = sorted_merged_log[['timestamp_numeric']].values
    dbscan = DBSCAN(eps=gamma, min_samples=min_samples, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    filtered_batches = []

    # Step 2: Iterate over clusters and filter each batch for FIFO
    for label in cluster_labels:
        batch = sorted_merged_log[sorted_merged_log["cluster"] == label].drop(columns="cluster")

        # Find longest increasing subsequence within each batch (FIFO filtering)
        batch = batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)

        valid_indices = find_largest_increasing_subsequence2(batch["start_time"].values)
        filtered_batch = batch.iloc[valid_indices]

        # If filtered batch is not empty, store it
        if len(filtered_batch) >= min_batch:
            filtered_batches.append(filtered_batch)

    # Step 3: Re-run DBSCAN on each filtered batch to check if traces are still close
    final_batches = []
    for batch in filtered_batches:
        if len(batch) < min_batch:  # Skip small batches
            continue

        # Convert timestamps again for DBSCAN
        batch = batch.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)


        batch['timestamp_numeric'] = batch['start_time'].apply(lambda x: int(x.timestamp()))

        X = batch[['timestamp_numeric']].values
        dbscan = DBSCAN(eps=gamma, min_samples=min_samples, metric='euclidean')
        batch['cluster'] = dbscan.fit_predict(X)

        for label in batch['cluster'].unique():
            if label == -1:  # Ignore noise
                continue

            final_batch = batch[batch["cluster"] == label].drop(columns=["cluster", "timestamp_numeric"])
            if len(final_batch) >= min_batch:
                final_batches.append(final_batch)

    return final_batches, sorted_merged_log


def find_largest_increasing_subsequence2(values):
    """
    Finds the longest non-strictly increasing subsequence in a list of values.
    
    :param values: List of numerical values.
    :return: Indices of the longest increasing subsequence.
    """
    n = len(values)
    if n == 0:
        return []

    # Stores the last index of the longest increasing subsequence of a given length
    subseq_end_at_length = [-1] * n
    # Stores the index of the predecessor of each element in the longest subsequence
    predecessors = [-1] * n
    # Length of the longest found subsequence
    length = 0

    for i in range(n):
        # Binary search to find the longest valid subsequence that can be extended
        low, high = 0, length
        while low < high:
            mid = (low + high) // 2
            if values[subseq_end_at_length[mid]] <= values[i]:  # Non-strictly increasing condition
                low = mid + 1
            else:
                high = mid
        
        # Update the tracking arrays
        if low > 0:
            predecessors[i] = subseq_end_at_length[low - 1]
        
        subseq_end_at_length[low] = i
        
        if low == length:
            length += 1

    # Reconstruct the longest increasing subsequence using predecessors
    sequence = []
    k = subseq_end_at_length[length - 1]
    while k != -1:
        sequence.append(k)
        k = predecessors[k]

    return sequence[::-1]  # Return in correct order


def Batch_on_start_fifo_DBSCAN(start_activity, end_activity, event_log, gamma = 0, min_batch = 20, min_samples = 2):
    batches = []
    traces_in_batch = []

    gamma = gamma * 86400
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    #sorted_merged_log['timestamp_numeric'] = pd.to_datetime(sorted_merged_log["end_time"]).astype('int64') / 10**9
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp())) #convert to unix seconds


    X = sorted_merged_log[['timestamp_numeric']].values

    #scaler = StandardScaler() #optional
    #X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=gamma, min_samples= min_samples, metric='euclidean')
    sorted_merged_log['cluster'] = dbscan.fit_predict(X)

    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    batches = [sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric') for label in cluster_labels]

    filtered_batches = []  # List to store only valid batches

    for label in cluster_labels:
        batch = sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric')
        traces_in_batch = []  # Temporary list to hold valid traces in the batch
    
        for i in range(len(batch) - 1):
            current_trace = batch.iloc[i]
            next_trace = batch.iloc[i + 1]
        
            # Add the first trace and initialize batch_start_time
            if i == 0:
                traces_in_batch.append(current_trace)
                batch_start_time = current_trace["end_time"]
        
            # Check FIFO condition
            if (
                current_trace["end_time"] <= next_trace["end_time"] and 
                current_trace["start_time"] <= next_trace["start_time"]
            ):
                traces_in_batch.append(next_trace)  # Valid, add to current batch
            else:
                continue
                # If condition fails, check if the current batch is large enough
                #if len(traces_in_batch) >= min_batch:
                #    filtered_batches.append(pd.DataFrame(traces_in_batch))
                # Reset for the next possible batch
                #traces_in_batch = [next_trace]
                #batch_start_time = next_trace["end_time"]

        # After the loop, check the last batch
        if len(traces_in_batch) >= min_batch:
            filtered_batches.append(pd.DataFrame(traces_in_batch))


    return filtered_batches, sorted_merged_log


def Batch_on_end_unordered(start_activity, end_activity, event_log, time_window = 0, min_batch_size = 20):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    #sorted_merged_log.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\test1.csv")
    
    for i in range(len(sorted_merged_log)-1):
        
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]

        if i == 0:
            print(type(current_trace["end_time"]))
            print(type(next_trace["end_time"]))
            print(type(time_window))
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if current_trace["end_time"] <= next_trace["end_time"] <= time_window + current_trace["end_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    return batches, sorted_merged_log

def new_Batch_on_end_unordered(start_activity, end_activity, event_log, time_window = 0, min_batch_size = 20):
    time_window = pd.Timedelta(days=time_window)  # Convert to timedelta
    temp_batches = []
    filtered_batches = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    
    # Prepare and sort the data
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    i = 0  # Index to track the starting trace
    while i < len(sorted_merged_log):
        first_trace = sorted_merged_log.iloc[i]
        traces_in_batch = [first_trace]
        batch_start_time = first_trace["end_time"]  # Reference start time for time window

        # Start batch formation
        for j in range(i + 1, len(sorted_merged_log)):
            next_trace = sorted_merged_log.iloc[j]

            # FIFO and time window condition
            if (
                first_trace["end_time"] <= next_trace["end_time"] <= batch_start_time + time_window
                # andfirst_trace["start_time"] <= next_trace["start_time"]
            ):
                traces_in_batch.append(next_trace)
            else:
                break  # Stop adding traces once conditions are violated

        # If batch meets the required size, save it and move index forward
        if len(traces_in_batch) >= min_batch_size:
            temp_batches.append(pd.DataFrame(traces_in_batch))
            i += len(traces_in_batch)  # Move index forward to next trace after the first trace
        else:
            i += 1  # Move index forward by 1 to check the next trace
    return temp_batches, sorted_merged_log

def new_Batch_on_end_fifo_3(start_activity, end_activity, event_log, time_window=0, min_batch_size=20):
    time_window = pd.Timedelta(days=time_window)  # Convert to timedelta
    temp_batches = []
    filtered_batches = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    
    # Prepare and sort the data
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    i = 0  # Index to track the starting trace
    while i < len(sorted_merged_log):
        first_trace = sorted_merged_log.iloc[i]
        traces_in_batch = [first_trace]
        batch_start_time = first_trace["end_time"]  # Reference start time for time window

        # Start batch formation
        for j in range(i + 1, len(sorted_merged_log)):
            next_trace = sorted_merged_log.iloc[j]

            # FIFO and time window condition
            if (
                first_trace["end_time"] <= next_trace["end_time"] <= batch_start_time + time_window
                # andfirst_trace["start_time"] <= next_trace["start_time"]
            ):
                traces_in_batch.append(next_trace)
            else:
                break  # Stop adding traces once conditions are violated

        # If batch meets the required size, save it and move index forward
        if len(traces_in_batch) >= min_batch_size:
            temp_batches.append(pd.DataFrame(traces_in_batch))
            i += len(traces_in_batch)  # Move index forward to next trace after the first trace
        else:
            i += 1  # Move index forward by 1 to check the next trace


    for batch in temp_batches:
        traces_in_batch = []  # Temporary list to hold valid traces in the batch
        end_time_indices = []
        start_time_indices = []

        batch = batch.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

        end_time_indices = find_largest_increasing_subsequence2(batch["end_time"].values)
        traces_in_batch = batch.iloc[end_time_indices]
        if len(traces_in_batch) >= min_batch_size:
            traces_in_batch = traces_in_batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)
            filtered_batches.append(traces_in_batch)
            
    return filtered_batches, sorted_merged_log
    


def Batch_on_end_fifo2(
    start_activity, end_activity, event_log, time_window=0, min_batch_size=20, fifo_violation_tolerance=0
):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    violations_in_batch = 0  # Track FIFO violations in the current batch
    
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    
    for i in range(len(sorted_merged_log) - 1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]

        if current_trace["end_time"] <= next_trace["end_time"] <= time_window + current_trace["end_time"]:
            if current_trace["start_time"] <= next_trace["start_time"]:  # FIFO condition
                traces_in_batch.append(next_trace)
            else:
                # Allow deviation for FIFO condition
                if violations_in_batch < fifo_violation_tolerance:
                    traces_in_batch.append(next_trace)
                    violations_in_batch += 1
                else:
                    if len(traces_in_batch) >= min_batch_size:
                        batches.append(pd.DataFrame(traces_in_batch))
                    traces_in_batch = [next_trace]
                    batch_start_time = next_trace["end_time"]
                    violations_in_batch = 0  # Reset violations for the new batch
        else:
            # Close the current batch and start a new one
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            batch_start_time = next_trace["end_time"]
            violations_in_batch = 0  # Reset violations for the new batch

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    
    return batches, sorted_merged_log

def Batch_on_start_fifo(start_activity, end_activity, event_log, time_window = 0, min_batch_size = 20):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="start")
    
    for i in range(len(sorted_merged_log)-1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        if current_trace["start_time"] <= next_trace["start_time"] <= time_window + current_trace["start_time"] and current_trace["end_time"] <= next_trace["end_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    

    batches_inds = set()
    for b in batches:
        batches_inds.update(b['case_ID'].values)


    #return event_log[event_log['case:concept:name'].isin(batches_inds)], batches
    return batches, sorted_merged_log

def Batch_on_start_unordered(start_activity, end_activity, event_log, time_window = 0, min_batch_size = 20):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="start")
    
    for i in range(len(sorted_merged_log)-1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        if current_trace["start_time"] <= next_trace["start_time"] <= time_window + current_trace["start_time"]:
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    

    batches_inds = set()
    for b in batches:
        batches_inds.update(b['case_ID'].values)


    #return event_log[event_log['case:concept:name'].isin(batches_inds)], batches
    return batches, sorted_merged_log

def Batch_on_end_lifo_old(start_activity, end_activity, event_log, min_batch_size = 20):
    
    
    batches = []
    traces_in_batch = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log_2 = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    #sorted_merged_log.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\segement5.csv")

    sorted_merged_log = sorted_merged_log_2.sort_values(by=['start_time', 'end_time'], ascending=[True, False])
    
    for i in range(len(sorted_merged_log)-1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if current_trace["end_time"] > next_trace["end_time"]  and current_trace["start_time"] < next_trace["start_time"]: 
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))

    return batches, sorted_merged_log_2

def new_Batch_on_start_unordered(start_activity, end_activity, event_log, min_batch_size, gamma):
    time_window = pd.Timedelta(days=time_window)  # Convert to timedelta
    temp_batches = []
    filtered_batches = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    
    # Prepare and sort the data
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="start")

    i = 0  # Index to track the starting trace
    while i < len(sorted_merged_log):
        first_trace = sorted_merged_log.iloc[i]
        traces_in_batch = [first_trace]
        batch_start_time = first_trace["start_time"]  # Reference start time for time window

        # Start batch formation
        for j in range(i + 1, len(sorted_merged_log)):
            next_trace = sorted_merged_log.iloc[j]

            # FIFO and time window condition
            if (
                first_trace["start_time"] <= next_trace["start_time"] <= batch_start_time + time_window
                # andfirst_trace["start_time"] <= next_trace["start_time"]
            ):
                traces_in_batch.append(next_trace)
            else:
                break  # Stop adding traces once conditions are violated

        # If batch meets the required size, save it and move index forward
        if len(traces_in_batch) >= min_batch_size:
            temp_batches.append(pd.DataFrame(traces_in_batch))
            i += len(traces_in_batch)  # Move index forward to next trace after the first trace
        else:
            i += 1  # Move index forward by 1 to check the next trace
    return temp_batches, sorted_merged_log

def new_Batch_on_start_fifo(start_activity, end_activity, event_log, min_batch_size, gamma):
    time_window = pd.Timedelta(days=time_window)  # Convert to timedelta
    temp_batches = []
    filtered_batches = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    
    # Prepare and sort the data
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="start")

    i = 0  # Index to track the starting trace
    while i < len(sorted_merged_log):
        first_trace = sorted_merged_log.iloc[i]
        traces_in_batch = [first_trace]
        batch_start_time = first_trace["start_time"]  # Reference start time for time window

        # Start batch formation
        for j in range(i + 1, len(sorted_merged_log)):
            next_trace = sorted_merged_log.iloc[j]

            # FIFO and time window condition
            if (
                first_trace["start_time"] <= next_trace["start_time"] <= batch_start_time + time_window
                # andfirst_trace["start_time"] <= next_trace["start_time"]
            ):
                traces_in_batch.append(next_trace)
            else:
                break  # Stop adding traces once conditions are violated

        # If batch meets the required size, save it and move index forward
        if len(traces_in_batch) >= min_batch_size:
            temp_batches.append(pd.DataFrame(traces_in_batch))
            i += len(traces_in_batch)  # Move index forward to next trace after the first trace
        else:
            i += 1  # Move index forward by 1 to check the next trace


    for batch in temp_batches:
        traces_in_batch = []  # Temporary list to hold valid traces in the batch
        end_time_indices = []
        start_time_indices = []

        batch = batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)

        end_time_indices = find_largest_increasing_subsequence2(batch["start_time"].values)
        traces_in_batch = batch.iloc[end_time_indices]
        if len(traces_in_batch) >= min_batch_size:
            traces_in_batch = traces_in_batch.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)
            filtered_batches.append(traces_in_batch)
            
    return filtered_batches, sorted_merged_log

        
def find_unique_activites(event_log):
    return event_log["concept:name"].unique()

def label_dfs(dfs):
    for i, df in enumerate(dfs):
        df['label'] = i
    return dfs


if __name__ == "__main__":

    event_log = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor_Thesis_expexperiments\\BPI Challenge 2017_1_all\\BPI Challenge 2017.xes.gz")
    print(event_log)
    pass