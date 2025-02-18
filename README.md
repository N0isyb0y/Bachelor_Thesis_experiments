# Bachelor_Thesis_experiments
Repository to recreate findings
# Mining-Patterns-in-Performance-Spectrum

Preperation steps:
1. install pipenv with pip install pipenv
2. Navigate to your project directory and create a virtual environment with pipenv install

running frontend:
1. enter pipenv run python main.py and then the app will be accessible at http://127.0.0.1:5000/
2. upload xes file
3. input the parameters (batch v1 = original, batch v2 = time window)
4. press submit selection and wait for visualization
extra: also possible to change the colors of quantiles of duration in the bottom chart in index.html line 309 and also the batched traces in the top chart in line 304

running metrics/analysis:
1. go into Batch_analysis.py and input the algorithms you want to analyze
2. choose the input parameters for the selected algorithms
3. in main(): make sure the correct event_log gets loaded (which is fitting to the input parameters)
4. receive the csv with the metrics

Running experiment in Thesis (tables for segments)
1. go into Batch_analysis.py and copy the input for the Segment you want to check (provided from line 39-74) [setup for S2,S3,S4 already given/ uncommented]
2. enter "pipenv run python Batch_analysis.py" into cmd

Running experiment in Thesis (Performance spectrum)
1. enter "pipenv run python main.py"
2. open http://127.0.0.1:5000/
3. upload the .xes file you want to check
4. input the two activites for the segment
5. input the algorithm and the parameters
6. Press submit selection
