import pandas as pd
from analysis import *
from pattern_miner import *
from pm4py.objects.log.importer.xes import importer as xes_importer
import time

# Define batch algorithms and corresponding function references
batch_algorithms = {
    #"original FIFO": Batch_on_end_fifo,
    "DBSCAN FIFO": Batch_on_end_fifo_DBSCAN2,
    #"time window FIFO": new_Batch_on_end_fifo_3,
    #"Batch_on_end_unordered": Batch_on_end_unordered,
    #"new_Batch_on_end_unordered": new_Batch_on_end_unordered,
    #"Batch_on_end_unordered_DBSCAN": Batch_on_end_unordered_DBSCAN,
    #"Sequence_DBSCAN": sequence_DBSCAN,
    #"LIFO": Batch_on_end_lifo
}

# Define test input parameters

input_parameters = [
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 1, "min_batch_size": 20, "gamma": 2},
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 1, "min_batch_size": 20, "gamma": 20},
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 5, "min_batch_size": 20, "gamma": 2},
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 5, "min_batch_size": 20, "gamma": 20},

    # Create Fine → Payment
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 1, "min_batch_size": 30, "gamma": 2},
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 1, "min_batch_size": 30, "gamma": 30},
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 2, "min_batch_size": 30, "gamma": 2},
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 2, "min_batch_size": 30, "gamma": 30},

    # Payment → Add Penalty
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 1, "min_batch_size": 20, "gamma": 2},
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 1, "min_batch_size": 20, "gamma": 20},
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 5, "min_batch_size": 20, "gamma": 2},
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 5, "min_batch_size": 20, "gamma": 20},


]
"""
input_parameters = [
    # Add Penalty → Payment
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 5, "min_batch_size": 20, "gamma": 0.1},

    # Create Fine → Payment
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Create Fine", "end_activity": "Payment", "time_window": 2, "min_batch_size": 20, "gamma": 0.1},

    # Payment → Add Penalty
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Payment", "end_activity": "Add penalty", "time_window": 5, "min_batch_size": 20, "gamma": 0.1},

    # Additional Activity Pairs
]


input_parameters = [
    # 2019: Vendor creates invoice → Record Goods Receipt
    {"start_activity": "Vendor creates invoice", "end_activity": "Record Goods Receipt", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Vendor creates invoice", "end_activity": "Record Goods Receipt", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Vendor creates invoice", "end_activity": "Record Goods Receipt", "time_window": 2, "min_batch_size": 20, "gamma": 0.1},

    # 2019: Record Goods Receipt → Record Invoice Receipt
    {"start_activity": "Record Goods Receipt", "end_activity": "Record Invoice Receipt", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Record Goods Receipt", "end_activity": "Record Invoice Receipt", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "Record Goods Receipt", "end_activity": "Record Invoice Receipt", "time_window": 2, "min_batch_size": 20, "gamma": 0.1},
    ]
"""
"""
input_parameters = [
    # 2017: W_Call incomplete files → O_Accepted
    {"start_activity": "W_Call incomplete files", "end_activity": "O_Accepted", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "W_Call incomplete files", "end_activity": "O_Accepted", "time_window": 1, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "W_Call incomplete files", "end_activity": "O_Accepted", "time_window": 2, "min_batch_size": 20, "gamma": 0.1},
]
"""
"""
input_parameters = [
    # 2018: mail income → mail valid
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 5, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 10, "min_batch_size": 20, "gamma": 0.1},
]

input_parameters = [
    {"start_activity": "Add penalty", "end_activity": "Payment", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
]
#data input with BF = 99%
input_parameters = [
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 0.000001, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 5, "min_batch_size": 20, "gamma": 0.1},
    {"start_activity": "mail income", "end_activity": "mail valid", "time_window": 10, "min_batch_size": 20, "gamma": 0.1},
]

input_parameters = []

activities_2018 = [
    "mail income", "mail valid", "initialize", "begin editing", "finish editing",
    "save", "insert document", "performed", "calculate", "decide", "begin payment",
    "abort payment", "finish payment", "revoke decision", "check",
    "remove document", "plan", "finish preparations", "begin preparations",
    "prepare external", "abort external", "prepare offline", "performed offline",
    "withdraw", "revoke withdrawal", "begin admissibility check",
    "check admissibility", "approve", "take original document", "refuse",
    "begin editing from refused", "revoke approval", "discard", "cancel offline",
    "change department", "create", "restart editing", "finish pre-check",
    "calculate protocol", "clear", "correction GFM17"
]

# Generate all possible start-end activity pairs
for i in range(len(activities_2018)):
    for j in range(i + 1, len(activities_2018)):  # Ensuring start != end
        input_parameters.append({
            "start_activity": activities_2018[i],
            "end_activity": activities_2018[j],
            "time_window": 0.000001,
            "min_batch_size": 20,
            "gamma": 0.1
        })

input_parameters = []

activities_2019 = [
    "Create Purchase Order Item",
    "Vendor creates invoice", "Record Goods Receipt", "Record Invoice Receipt",
    "Clear Invoice", "Record Service Entry Sheet", 
]

# Generate all consecutive start-end activity pairs
for i in range(len(activities_2019) - 1):
    input_parameters.append({
        "start_activity": activities_2019[i],
        "end_activity": activities_2019[i + 1],
        "time_window": 0.000001,
        "min_batch_size": 20,
        "gamma": 0.1
    })
"""




def main():
    print("Script started...")  # Debugging start message
    
    # Load event log
    #traffic
    event_log = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\\Mining-Patterns-in-Performance-Spectrum\\Road_Traffic_Fine_Management_Process.xes")
    #2017
    #event_log = import_xes("C:\\Users\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2017_1_all\\BPI Challenge 2017.xes\\BPI Challenge 2017.xes")
    #2018
    #event_log = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2018_1_all\\BPI Challenge 2018.xes\\BPI Challenge 2018.xes")
    #2019
    #event_log = import_xes("C:\\Users\\Maurice\\Desktop\\Bachelor thesis new\\BPI Challenge 2019_1_all\\BPI_Challenge_2019.xes")
    dataset = []

    print("Starting batch processing...")
    for algo_name, algo_func in batch_algorithms.items():
        print(f"Running {algo_name}...")  # Debug: Check which algorithm runs
        for params in input_parameters:
            print(f"Using parameters: {params}")  # Debug: Check parameters used
            start_time = time.time()
            try:
                # Call the correct function dynamically
                if algo_name == "original FIFO":
                    batches, processed_log = Batch_on_end_fifo(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "time window FIFO":
                    batches, processed_log = new_Batch_on_end_fifo_3(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "DBSCAN FIFO":
                    batches, processed_log = Batch_on_end_fifo_DBSCAN2(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"],
                        params["gamma"]
                    )
                    end_time = time.time()
                elif algo_name == "Batch_on_end_unordered":
                    batches, processed_log = Batch_on_end_unordered(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "new_Batch_on_end_unordered":
                    batches, processed_log = new_Batch_on_end_unordered(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "Batch_on_end_unordered_DBSCAN":
                    batches, processed_log = Batch_on_end_unordered_DBSCAN(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "Sequence_DBSCAN":
                    batches, processed_log = sequence_DBSCAN(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["time_window"], 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                elif algo_name == "LIFO":
                    batches, processed_log = Batch_on_end_lifo_old(
                        params["start_activity"], 
                        params["end_activity"], 
                        event_log, 
                        params["min_batch_size"]
                    )
                    end_time = time.time()
                else:
                    raise ValueError(f"Unknown algorithm: {algo_name}")
                execution_time = end_time - start_time
                
                print(f"{algo_name} completed successfully.")

                # Debugging print for batch structure
                print(f"Batches Type: {type(batches)}")
                print(f"First batch type: {type(batches[0]) if batches else 'Empty'}")
                print(f"First batch length: {len(batches[0]) if batches else 'Empty'}")

                if not batches:  # Check if batches list is empty
                    print(f"⚠️ No valid batches found for {algo_name} with parameters {params}")
                    continue
                # Aggregate metrics
                concat_df = pd.concat(batches, ignore_index=True)
                print(f"Batch size: {len(concat_df)}")  # Debug: Check batch size

                metrics = {
                    "number_of_traces": number_of_traces(concat_df),
                    "number_of_batches": number_of_batches(batches),
                    "batch_frequency": round(batch_frequency(processed_log, batches), 2),
                    #"batch_duration_max": round(batch_duration_max(batches), 2),
                    "average_batch_arrival_window": round(average_batch_arrival_window(batches), 2),
                    "average_batch_interval": round(average_batch_interval(batches), 2),
                    "average_batch_size": round(calculate_average_batch_size(batches),2),
                    "runtime": round(execution_time, 2)
                }


                dataset.append({
                    "batch_algorithm": algo_name,
                    "start_activity": params["start_activity"],
                    #"end_activity": params["end_activity"],
                    "time_window": params["time_window"],
                    "min_batch_size": params["min_batch_size"],
                    "gamma": params["gamma"],
                    **metrics
                })
                print(f"Metrics recorded for {algo_name}.")

            except Exception as e:
                import traceback
                print(f"Error running {algo_name} with params {params}: {e}")
                traceback.print_exc()  # Prints the full stack trace

    # Convert dataset into a DataFrame
    df_metrics = pd.DataFrame(dataset)

    # Save to CSV
    output_csv = "batch_comparison_results.csv"
    if len(df_metrics) > 0:
        print(f"Saving dataset to {output_csv}...")
        df_metrics.to_csv(output_csv, index=False)
        print("CSV saved successfully!")
    else:
        print("No data to save. CSV not created.")

# Ensure script runs when executed directly
if __name__ == "__main__":
    main()

