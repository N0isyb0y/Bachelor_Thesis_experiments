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
def custom_distance(c1, c2):
    """
    Custom distance function that checks a specific constraint
    and calculates a distance metric based on start and end times.
    """
    # Extract start and end timestamps
    #print(c1, c2)
    s1, e1 = c1
    s2, e2 = c2

    # Check the constraint: (s1 - s2) * (e1 - e2) >= 0
    if (s1 - s2) * (e1 - e2) < 0:
        return np.inf  # Assign a large distance if the constraint is violated

    # Calculate the absolute distance based on end times
    return np.abs(e1 - e2)

def compute_distance_matrix(data):
    LARGE_DISTANCE = 1e300
    n = len(data)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Extract features for both points
            s1, e1 = data[i]
            s2, e2 = data[j]

            # Apply constraint check
            if (s1 - s2) * (e1 - e2) < 0:
                dist = LARGE_DISTANCE
            else:
                dist = np.abs(e1 - e2)

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric matrix

    return dist_matrix

def compute_distance_matrix_LIFO(data):
    LARGE_DISTANCE = 1e300
    n = len(data)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Extract features for both points
            s1, e1 = data[i]
            s2, e2 = data[j]

            # Apply constraint check
            if (s1 - s2) * (e1 - e2) < 0:
                dist = 0
            else:
                dist = np.inf

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric matrix

    return dist_matrix

def lifo_condition(s1, e1, s2, e2):
    """Check if two lines cross based on the LIFO condition."""
    return (s1 - s2) * (e1 - e2) < 0

def strict_lifo_clustering(data):
    n = len(data)
    adjacency_matrix = np.zeros((n, n), dtype=bool)

    # Compute adjacency matrix based on LIFO condition
    for i in range(n):
        for j in range(i + 1, n):
            s1, e1 = data[i]
            s2, e2 = data[j]
            if lifo_condition(s1, e1, s2, e2):
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True

    # Find cliques (all points mutually connected)
    clusters = []
    for size in range(2, n + 1):  # Consider cliques of size 2 to n
        for combination in combinations(range(n), size):
            if all(adjacency_matrix[i, j] for i, j in combinations(combination, 2)):
                clusters.append([data[i] for i in combination])

    # Remove subsets of larger clusters
    filtered_clusters = []
    for cluster in clusters:
        if not any(set(cluster).issubset(set(other)) and cluster != other for other in clusters):
            filtered_clusters.append(cluster)

    return filtered_clusters

def Batch_on_end_lifo_DBSCAN_new2(start_activity, end_activity, event_log, gamma=0.0001, min_samples=20):
    gamma = gamma * 86400  # Convert gamma to seconds
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['start_timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp()))
    sorted_merged_log['end_timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp()))

    # Prepare data for clustering
    X = sorted_merged_log[['start_timestamp_numeric', 'end_timestamp_numeric']].values

    res = strict_lifo_clustering(X)

    return res


def Batch_on_end_lifo_DBSCAN_new(start_activity, end_activity, event_log, gamma=0.0001, min_samples=20):
    gamma = gamma * 86400  # Convert gamma to seconds
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['start_timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp()))
    sorted_merged_log['end_timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp()))

    # Prepare data for clustering
    X = sorted_merged_log[['start_timestamp_numeric', 'end_timestamp_numeric']].values

    dist_matrix = compute_distance_matrix_LIFO(X)

    # Define the custom metric for DBSCAN
    #def custom_metric(p1, p2):
    #    return custom_distance(p1, p2)

    # Apply DBSCAN with the custom distance metric
    dbscan = DBSCAN(eps=gamma, min_samples=2, metric="precomputed")
    sorted_merged_log['cluster'] = dbscan.fit_predict(dist_matrix)

    # Get unique cluster labels
    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    batches = []
    
    for label in cluster_labels:
        cluster_data = sorted_merged_log[sorted_merged_log['cluster'] == label]

        # If the cluster size is larger than the minimum size, keep it
        if len(cluster_data) >= min_samples:
            batches.append(cluster_data.drop(columns=['start_timestamp_numeric', 'end_timestamp_numeric']))

    return batches


def Batch_on_end_fifo_DBSCAN_new(start_activity, end_activity, event_log, gamma=0.0001, min_samples=20):
    gamma = gamma * 86400  # Convert gamma to seconds
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['start_timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp()))
    sorted_merged_log['end_timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp()))

    # Prepare data for clustering
    X = sorted_merged_log[['start_timestamp_numeric', 'end_timestamp_numeric']].values

    dist_matrix = compute_distance_matrix(X)

    # Define the custom metric for DBSCAN
    #def custom_metric(p1, p2):
    #    return custom_distance(p1, p2)

    # Apply DBSCAN with the custom distance metric
    dbscan = DBSCAN(eps=gamma, min_samples=2, metric="precomputed")
    sorted_merged_log['cluster'] = dbscan.fit_predict(dist_matrix)

    # Get unique cluster labels
    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    batches = []
    
    for label in cluster_labels:
        cluster_data = sorted_merged_log[sorted_merged_log['cluster'] == label]

        # If the cluster size is larger than the minimum size, keep it
        if len(cluster_data) >= min_samples:
            batches.append(cluster_data.drop(columns=['start_timestamp_numeric', 'end_timestamp_numeric']))

    return batches

def Batch_on_end_fifo_DBSCAN_new_fifo_filtered_after(start_activity, end_activity, event_log, gamma=0.0001, min_samples=20):
    gamma = gamma * 86400  # Convert gamma to seconds
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    # Convert timestamps to UNIX seconds
    sorted_merged_log['start_timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp()))
    sorted_merged_log['end_timestamp_numeric'] = sorted_merged_log['end_time'].apply(lambda x: int(x.timestamp()))

    # Prepare data for clustering
    X = sorted_merged_log[['start_timestamp_numeric', 'end_timestamp_numeric']].values

    dist_matrix = compute_distance_matrix(X)

    # Define the custom metric for DBSCAN
    #def custom_metric(p1, p2):
    #    return custom_distance(p1, p2)

    # Apply DBSCAN with the custom distance metric
    dbscan = DBSCAN(eps=gamma, min_samples=2, metric="precomputed")
    sorted_merged_log['cluster'] = dbscan.fit_predict(dist_matrix)

    # Get unique cluster labels
    cluster_labels = sorted_merged_log['cluster'].unique()
    cluster_labels = cluster_labels[cluster_labels != -1]  # Exclude noise points (-1)

    # Create a list of dataframes, each corresponding to one cluster
    batches = []
    filtered_batches = []

    for label in cluster_labels:
        batch = sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns=['start_timestamp_numeric', 'end_timestamp_numeric'])
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
                # If condition fails, check if the current batch is large enough
                if len(traces_in_batch) >= min_samples:
                    filtered_batches.append(pd.DataFrame(traces_in_batch))
                # Reset for the next possible batch
                traces_in_batch = [next_trace]
                batch_start_time = next_trace["end_time"]

        # After the loop, check the last batch
        if len(traces_in_batch) >= min_samples:
            filtered_batches.append(pd.DataFrame(traces_in_batch))


    return filtered_batches, sorted_merged_log

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

def Batch_on_start_unordered_DBSCAN(start_activity, end_activity, event_log, gamma = 0.0001, min_samples = 20):
    gamma = gamma * 86400
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")

    #sorted_merged_log['timestamp_numeric'] = pd.to_datetime(sorted_merged_log["end_time"]).astype('int64') / 10**9
    sorted_merged_log['timestamp_numeric'] = sorted_merged_log['start_time'].apply(lambda x: int(x.timestamp())) #convert to unix seconds


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
        if len(cluster_data) >= min_samples:
            batches.append(cluster_data.drop(columns='timestamp_numeric'))
    return batches, sorted_merged_log


def Batch_on_end_fifo_DBSCAN(start_activity, end_activity, event_log, gamma = 0.000001, min_batch = 20, min_samples = 2):
    batches = []
    traces_in_batch = []

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
    batches = [sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric') for label in cluster_labels]

    filtered_batches = []  # List to store only valid batches

    for label in cluster_labels:
        batch = sorted_merged_log[sorted_merged_log['cluster'] == label].drop(columns='timestamp_numeric')
        traces_in_batch = []  # Temporary list to hold valid traces in the batch
        end_time_indices = []
        start_time_indices = []

        batch = batch.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

        end_time_indices = find_largest_increasing_subsequence(batch["end_time"].values)
        traces_in_batch = batch.iloc[end_time_indices]
        if len(traces_in_batch) >= min_batch:
            traces_in_batch = traces_in_batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)
            filtered_batches.append(traces_in_batch)



# Intersect both sequences

        """
        for i in range(len(batch) - 1):
            current_trace = batch.iloc[i]
            next_trace = batch.iloc[i + 1]
            
            if next_trace["end_time"] >= current_trace["end_time"]:
                end_time_indices.append(i)
            #if next_trace["start_time"] >= current_trace["start_time"]:
            #    start_time_indices.append(i)
        last_trace = batch.iloc[-1]
        second_last_trace = batch.iloc[-2]  # Previous trace before the last one

        if last_trace["end_time"] >= second_last_trace["end_time"]:
            end_time_indices.append(len(batch) - 1)  # Append last trace


        largest_ascending_end_time_indices = find_largest_consecutive_ascending_sequence(end_time_indices, batch["end_time"].values)
        #final_indices = list(set(start_time_indices) & set(largest_ascending_end_time_indices))
        traces_in_batch = batch.iloc[largest_ascending_end_time_indices]

        if len(traces_in_batch) >= min_batch:
            filtered_batches.append(traces_in_batch)
        """
        """
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
        """


    return filtered_batches, sorted_merged_log

def Batch_on_end_fifo_DBSCAN2(start_activity, end_activity, event_log, gamma=0.000001, min_batch=20, min_samples=20):
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


def find_largest_increasing_subsequence(values):
        n = len(values)
        lengths = [1] * n
        predecessors = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if values[i] >= values[j] and lengths[i] < lengths[j] + 1:
                    lengths[i] = lengths[j] + 1
                    predecessors[i] = j

        max_length_idx = max(range(n), key=lambda i: lengths[i])

        sequence = []
        while max_length_idx != -1:
            sequence.append(max_length_idx)
            max_length_idx = predecessors[max_length_idx]

        return list(reversed(sequence))

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

def find_largest_increasing_subsequence3(values):
    """
    Finds the longest non-strictly increasing subsequence in a list of values.
    If multiple valid subsequences exist, the one with the closest indices is chosen.
    
    :param values: List of numerical values.
    :return: Indices of the longest increasing subsequence with the smallest range.
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

    # Dictionary to store multiple valid subsequences
    subsequences = {}

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

        # Store all possible subsequences
        sequence = []
        k = i
        while k != -1:
            sequence.append(k)
            k = predecessors[k]
        sequence.reverse()
        
        subsequences[tuple(sequence)] = sequence[-1] - sequence[0]  # Store range size

    # Select the sequence with the smallest range (closest indices)
    best_sequence = min(subsequences, key=subsequences.get)
    
    return list(best_sequence)




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
            first_trace = sorted_merged_log.iloc[i]
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if first_trace["end_time"] <= next_trace["end_time"] <= time_window + first_trace["end_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            first_trace = next_trace
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    return batches, sorted_merged_log

def new_Batch_on_end_fifo(start_activity, end_activity, event_log, time_window=0, min_batch_size=20):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    
    for i in range(len(sorted_merged_log) - 1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        
        if i == 0:
            first_trace = current_trace
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
        
        # Check both time window and FIFO conditions
        if (
            first_trace["end_time"] <= next_trace["end_time"] <= time_window + first_trace["end_time"] and
            first_trace["start_time"] <= next_trace["start_time"]
        ):
            traces_in_batch.append(next_trace)
        else:
            # Close the current batch if it meets the minimum batch size
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            # Start a new batch
            traces_in_batch = [next_trace]
            first_trace = next_trace
            batch_start_time = next_trace["end_time"]

    # Final check for the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    
    return batches, sorted_merged_log

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

        end_time_indices = find_largest_increasing_subsequence(batch["end_time"].values)
        traces_in_batch = batch.iloc[end_time_indices]
        if len(traces_in_batch) >= min_batch_size:
            traces_in_batch = traces_in_batch.sort_values(by=["end_time", "start_time"]).reset_index(drop=True)
            filtered_batches.append(traces_in_batch)
            
    return filtered_batches, sorted_merged_log



def Batch_on_end_fifo(start_activity, end_activity, event_log, time_window = 0, min_batch_size = 20, fifo_violation_tolerance=0):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    violations_in_batch = 0  # Track FIFO violations in the current batch
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    #sorted_merged_log.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\segement5.csv")
    
    for i in range(len(sorted_merged_log)-1):
        current_trace = sorted_merged_log.iloc[i]
        next_trace = sorted_merged_log.iloc[i + 1]
        if i == 0:
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if current_trace["end_time"] <= next_trace["end_time"] <= time_window + current_trace["end_time"] and current_trace["start_time"] <= next_trace["start_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if violations_in_batch < fifo_violation_tolerance:
                    traces_in_batch.append(next_trace)
                    violations_in_batch += 1
            else:
                if len(traces_in_batch) >= min_batch_size:
                    batches.append(pd.DataFrame(traces_in_batch))
                traces_in_batch = [next_trace]
                batch_start_time = next_trace["end_time"]
                violations_in_batch = 0  # Reset violations for the new batch
        #maybe add "else" to close current batch

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))
    return batches, sorted_merged_log


def Batch_on_end_fifo(
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

def Batch_on_end_lifo(start_activity, end_activity, event_log, time_window=0, min_batch_size=20):
    time_window = pd.Timedelta(days=time_window)
    batches = []
    traces_in_batch = []
    
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    
    i = 0  # Index to track the starting trace
    while i < len(sorted_merged_log):
        first_trace = sorted_merged_log.iloc[i]
        traces_in_batch = [first_trace]
        batch_start_time = first_trace["end_time"]  # Reference time for the batch

        # Start batch formation
        for j in range(i + 1, len(sorted_merged_log)):
            next_trace = sorted_merged_log.iloc[j]

            # LIFO condition: next trace must have an earlier or equal end_time and later start_time
            if first_trace["end_time"] > next_trace["end_time"] and first_trace["start_time"] < next_trace["start_time"]:
                traces_in_batch.append(next_trace)
            else:
                # If the LIFO condition does not hold, continue checking the next trace
                continue

        # If batch meets the required size, save it and move index forward
        if len(traces_in_batch) >= min_batch_size:
            batches.append(pd.DataFrame(traces_in_batch))
            i += len(traces_in_batch)  # Move index forward to next trace after the first trace
        else:
            i += 1  # Move index forward by 1 to check the next trace

    return batches, sorted_merged_log

def sequence_pattern(start_activity, end_activity, event_log, min_sequence_size = 10, time_window = 0, gamma = 0): #gamma = slope deviation, time window = timewindow allowed for next start
    gamma = pd.Timedelta(days=gamma)
    time_window = pd.Timedelta(days=time_window)
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    temp_sequence_size = 0
    first_duration = sorted_merged_log["duration"].iloc[0]

    temp_sequence = []
    sequence = []
    mult_sequences = [] #list with dfs 
    sorted_merged_log.iloc["checked"] = 0
    middle_trace = sorted_merged_log.iloc[0]

    for i in range(len(sorted_merged_log)-1):
        current_trace = sorted_merged_log.iloc[i]
        if current_trace["checked"] == 1: #if already considered skip
            continue
        if len(middle_trace) == 0:
            middle_trace = current_trace
            temp_sequence = [current_trace]
            possible_successor.iloc[j]["checked"] = 1
            temp_sequence_size += 1

        possible_successor = sorted_merged_log[(sorted_merged_log["duration"] <= first_duration + gamma) & (sorted_merged_log["duration"] >= first_duration - gamma) 
                                               & (sorted_merged_log["start_time"] <= middle_trace["start_time"] + first_duration + time_window) & (sorted_merged_log["start_time"] >= middle_trace["end_time"] + first_duration - time_window)]

        if possible_successor.empty == False:
            for j in range(len(possible_successor)):
                    temp_sequence.append(possible_successor.iloc[j])
                    possible_successor.iloc[j]["checked"] = 1

                    if len(possible_successor) % 2 == 1: #take median 
                        middle_trace = possible_successor[len(possible_successor)//2]

                    elif len(possible_successor) % 2 == 0: #can be changed to fixed middle_trace = possible_successor[len(possible_successor)//2]
                        temp1_time = possible_successor[len(possible_successor)//2]["start_time"]
                        temp2_time = possible_successor[len(possible_successor)//2 + 1]["start_time"]
                        middle_trace = possible_successor[len(possible_successor)//2]
                        middle_trace["start_time"] = (temp1_time + temp2_time)//2

            temp_sequence_size += 1

        elif possible_successor.empty == True:
            if temp_sequence_size >= min_sequence_size:
                sequence.append(pd.DataFrame(temp_sequence))
                mult_sequences.append(sequence)
            sequence = []
            middle_trace = []
            temp_sequence =[]
            
            
    return  mult_sequences, sorted_merged_log

def new_Batch_on_start_unordered(start_activity, end_activity, event_log, min_batch_size, gamma):
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
            first_trace = sorted_merged_log.iloc[i]
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if first_trace["start_time"] <= next_trace["start_time"] <= time_window + first_trace["start_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            first_trace = next_trace
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))

    return batches, sorted_merged_log

def new_Batch_on_start_fifo(start_activity, end_activity, event_log, min_batch_size, gamma):
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
            first_trace = sorted_merged_log.iloc[i]
            traces_in_batch.append(current_trace)
            batch_start_time = current_trace["end_time"]
            
        #if (pms_df.iloc[row_ind,3] <= pms_df.iloc[row_ind+1,3] <= datetime.timedelta(minutes=delta_min) + pms_df.iloc[row_ind,3]) and (pms_df.iloc[row_ind+1,1] >= pms_df.iloc[row_ind,1]):
        if first_trace["start_time"] <= next_trace["start_time"] <= time_window + first_trace["start_time"]: #check if conditions 1. in batch 2. not crossing
            traces_in_batch.append(next_trace)
        else:
            if len(traces_in_batch) >= min_batch_size:
                batches.append(pd.DataFrame(traces_in_batch))
            traces_in_batch = [next_trace]
            first_trace = next_trace
            batch_start_time = next_trace["end_time"]

    # Check the last batch
    if len(traces_in_batch) >= min_batch_size:
        batches.append(pd.DataFrame(traces_in_batch))

    return batches, sorted_merged_log


def duration(start_activity, end_activity, event_log, min_size = 10, gamma = 0): #gamma = slope deviation, time window = timewindow allowed for next start
    #gamma = pd.Timedelta(days=gamma)
    filtered_event_log = event_log[["time:timestamp", "case:concept:name", "concept:name"]]
    sorted_merged_log = prepare_data(event_log=filtered_event_log, start_activity=start_activity, end_activity=end_activity, batch_type="end")
    #print(sorted_merged_log)

    
    sorted_merged_log['duration_days'] = sorted_merged_log['duration'].dt.days
    """
    # Get all unique durations in the 'duration_days' column
    unique_durations = sorted_merged_log['duration_days'].unique()

    # Dictionary to store DataFrames for each duration
    duration_dfs_list = []

    # Loop over each unique duration value and filter the DataFrame
    for duration in unique_durations:
        # Append the filtered DataFrame to the list
        duration_dfs_list.append(sorted_merged_log[sorted_merged_log['duration_days'] == duration])
            
    print(duration_dfs_list[0])
    print(duration_dfs_list[1])
    """
    unique_durations = sorted(sorted_merged_log['duration_days'].unique())

    # List to store DataFrames for each duration group with a time window
    duration_dfs_list = []

    # Keep track of processed indices so that we don't double count overlapping windows
    processed_indices = set()

    # Loop over each unique duration value
    for duration in unique_durations:
        # Define the lower and upper bounds for the time window
        lower_bound = duration - gamma
        upper_bound = duration + gamma
    
        # Filter the DataFrame to include durations within the time window range
        filtered_df = sorted_merged_log[(sorted_merged_log['duration_days'] >= lower_bound) & (sorted_merged_log['duration_days'] <= upper_bound)]

        #Avoid duplicate groups by checking if the rows were already processed
        filtered_df = filtered_df[~filtered_df.index.isin(processed_indices)]

        #print(filtered_df)
    
        # If there are any valid rows left, append them to the list
        if len(filtered_df) >= min_size:
            duration_dfs_list.append(filtered_df)
            # Add the indices of the processed rows to the set
            processed_indices.update(filtered_df.index)
    return  duration_dfs_list, sorted_merged_log
        
def mine_patterns(event_log, start_activity, end_activity, time_window = 0, min_batch_size = 20, min_sequence_size = 10, batch_type = "end", fifo = True):
    if fifo:
        if batch_type == "start":
            return Batch_on_start_fifo(start_activity, end_activity, event_log, time_window, min_batch_size)
        elif batch_type == "end":
            return Batch_on_end_fifo(start_activity, end_activity, event_log, time_window, min_batch_size)
    elif not fifo:
        if batch_type == "start":
            return Batch_on_start_unordered(start_activity, end_activity, event_log, time_window, min_batch_size)
        elif batch_type == "end":
            return Batch_on_end_unordered(start_activity, end_activity, event_log, time_window, min_batch_size)

def find_unique_activites(event_log):
    return event_log["concept:name"].unique()

def label_dfs(dfs):
    for i, df in enumerate(dfs):
        df['label'] = i
    return dfs


if __name__ == "__main__":

    #event_log = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\\Mining-Patterns-in-Performance-Spectrum\\Road_Traffic_Fine_Management_Process.xes")
    #event_log_2017 = import_xes("C:\\Users\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2017_1_all\\BPI Challenge 2017.xes\\BPI Challenge 2017.xes")
    #event_log_2018 = import_xes("C:\\Users\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2018_1_all\\BPI Challenge 2018.xes\\BPI Challenge 2018.xes")
    #event_log_2019 = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2019_1_all\\BPI_Challenge_2019.xes")
    #event_log_2020_international = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\BPI Challenge 2020_ International Declarations_1_all\\InternationalDeclarations.xes\\InternationalDeclarations.xes")

    
    
    #event_log.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\startdata.csv")
    #print(event_log)
    #batches = Batch_on_end_unordered("Add penalty", "Payment", event_log, 0)
    #batches = Batch_on_end_lifo("Add penalty", "Payment", event_log, 3)

    #batches = Batch_on_end_fifo_DBSCAN_new("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)

    #batches,useless = new_Batch_on_end_fifo_3("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)
    #print(batches)
    #print(type(batches[0]))
    #print(batches[0])
    #print(len(batches[0]))
    
    #batches = Batch_on_end_fifo_DBSCAN_new("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)
    #batches2,a = Batch_on_end_fifo_DBSCAN_new2("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)
    #batches3,b = Batch_on_end_fifo_DBSCAN("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)
    #batches4,b = Batch_on_end_unordered_DBSCAN("Add penalty", "Send Appeal to Prefecture", event_log, 2, 20)
    #batches5 = Batch_on_end_lifo_DBSCAN_new("Add penalty", "Send Appeal to Prefecture", event_log, 2, 4)
    #batches6 = Batch_on_end_lifo_DBSCAN_new2("Add penalty", "Send Appeal to Prefecture", event_log, 2, 4)
    """
    batches7, a = Batch_on_end_unordered("Add penalty", "Send Appeal to Prefecture", event_log, 0, 20)
    concat_df = pd.concat(batches7, ignore_index=True)
    print(batches7)
    metrics = {}
    if isinstance(batches7, list) and all(isinstance(df, pd.DataFrame) for df in batches7):
        metrics["number_of_traces"] = number_of_traces(concat_df)
        metrics["number_of_batches"] = number_of_batches(batches7)
        metrics["batch_frequency"] = batch_frequency(a, batches7)
        metrics["batch_duration_max"] = batch_duration_max(batches7)
        metrics["average_batch_arrival_window"] = average_batch_arrival_window(batches7)
        metrics["average_batch_interval"] = average_batch_interval(batches7)

    print(metrics)
    """
    #print([len(b) for b in batches]) 
    #print([len(b) for b in batches2]) 
    #print([len(b) for b in batches3]) 
    #print([len(b) for b in batches4])
    #for batch in batches5:
    #    print(batch)
    #print([len(b) for b in batches5])
    #print(batches[5])
    #print(batches4[5])
    #batches_in_eventlog1 = pd.concat(batches, ignore_index=True)
    #batches_in_eventlog1.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\DBSCAN_NEW.csv")




    #batches, event_log_2 = Batch_on_end_unordered_DBSCAN("Add penalty", "Payment", event_log, 1, 20)
    #batches_2, event_log_3 = Batch_on_end_unordered("Add penalty", "Payment", event_log, 1, 20)
    #print(average_batch_arrival_window(batches))
    #print(average_batch_arrival_window(batches))
    #print(average_batch_interval(batches))
    #print(number_of_batches(batches))
    #print(number_of_traces(event_log_2))
    #for b in batches:
    #    print(b)

    #print([len(b) for b in batches]) 

    #print(len(batches))


    #print(batches[2])

    #batches = Batch_on_end_fifo("Add penalty", "Payment", event_log, 300, 20)
    #durations = duration("Create Fine", "Send Fine", event_log, 2400, 1)
    #batches = mine_patterns(event_log, "Payment", "Add penalty", 1, 20, 10, "end", False)
    #batches2 = new_Batch_on_end_unordered("Payment", "Add penalty", event_log, 2, 20)
    #for idx, duration in enumerate(durations):
    #    print(f"duration {idx + 1}:")
    #    print(duration)
    #    print()
    #final = label_dfs(durations)
    #for duration in final:
    #    print(duration)
    #    print()
    #print(len(final))
    #print(type(find_unique_activites(event_log)))
    #start_time = time.time()



    """
    batches, useless = Batch_on_end_fifo_DBSCAN2("Add penalty", "Payment", event_log, 0.0001, 20)

    batches4, useless = Batch_on_end_fifo_DBSCAN("Add penalty", "Payment", event_log, 0.0001, 20)
    
    batches2, useless = Batch_on_end_fifo_DBSCAN("Add penalty", "Payment", event_log, 1, 20)

    batches3, useless = Batch_on_end_fifo_DBSCAN2("Add penalty", "Payment", event_log, 1, 20)

    temp = 0
    for batch in batches:
        temp = temp + len(batch)
    print(temp)
    print(len(batches))

    temp4 = 0
    for batch in batches4:
        temp4 = temp4 + len(batch)
    print(temp4)
    print(len(batches4))

    temp2 = 0
    for batch in batches2:
        temp2 = temp2 + len(batch)
    print(temp2)
    print(len(batches2))

    temp3 = 0
    for batch in batches3:
        temp3 = temp3 + len(batch)
    print(temp3)
    print(len(batches3))

    print([len(b) for b in batches])
    print([len(b) for b in batches2])
    print([len(b) for b in batches3])
    """
    #print(batches[0])
    #print(batches2[3])
    #print(batches3[2])

    #print(batches2[0])

    #values = [5, 4, 3, 2, 1, 2]
    #print(find_largest_increasing_subsequence(values))
    #print(batches[0])
    #print(batches2[0])
    #print(batches3[0])
    #batches = Batch_on_end_unordered("Payment", "Send for Credit Collection", event_log, 0, 20)
    #batches = Batch_on_end_lifo("Add penalty", "Payment", event_log, 3)
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print(f"Execution time: {execution_time} seconds")
    #batches_in_eventlog, batches = Batch_on_start_fifo("Create Fine", "Payment", event_log, 0, 30)
    #batches_in_eventlog.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\start_fifo_create fine_payment.csv")
    #batches_in_eventlog2 = pd.concat(batches, ignore_index=True)
    #batches_in_eventlog2.to_csv("C:\\Users\\Maurice\\Desktop\\test csvs\\start_fifo_create fine_paymentfinal.csv")
    #batches2 = Batch_on_end_fifo("Payment", "Send for Credit Collection", filtered_event_log, 0,20)

    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_rows', 2)
    
    #print([len(b) for b in batches])
    #print("")
    #print(f"n_batches:{len(batches2)}")
    #print([len(b) for b in batches2])
    #print(batches[0])

    #for idx, batch in enumerate(batches):
    #    print(f"Batch {idx + 1}:")
    #    print(batch)
    #    print()
    #df = []
    #batches_concat = pd.concat(batches, ignore_index=True)
    #batches_concat.to_csv("C:\\Users\\Maurice\\Desktop\\test_outputs\\new.csv")
    
    """ #bring into original format
    for batch in batches:
        df_x = batch[['case:concept:name', 'time:timestamp_x', 'concept:name_x']].rename(columns={
                    'time:timestamp_x': 'time:timestamp',
                    'concept:name_x': 'concept:name'})
        df_y = batch[['case:concept:name', 'time:timestamp_y', 'concept:name_y']].rename(columns={
                    'time:timestamp_y': 'time:timestamp',
                    'concept:name_y': 'concept:name'})  

        df.append(pd.concat([df_x, df_y], ignore_index=True))
    df_merged = pd.concat(df, ignore_index=True)
    event_log2 = pm4py.format_dataframe(df_merged, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    """
    #df_test.to_csv("c:\\Users\\Maurice\\Desktop\\Bachelor thesis\\test21.csv")
    #pm4py.view_performance_spectrum(event_log2, ["Payment","Send for Credit Collection"], format="png")
    #pd.set_option('display.max_rows', None)
    #print(event_log2)
    #pm4py.view_performance_spectrum(event_log, ["Create Fine","Payment"], format="png")

