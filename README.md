# README

This project contains the implementation of CritMat, which is based on the paper: https://www.cs.utexas.edu/~yzhang/papers/critmat-dsn05.pdf

The main simulator is in test.py. In order to run it, you will need to clone the following dependencies:

1) https://github.com/minyee/facebook_dcn_traffic
2) https://github.com/minyee/robust_topology_engineering

Please make sure that you are cloning these two dependencies and CritMat into the same base directory. This is important for the simulator to run smoothly. For instance, all three projects must have the same base directory called <BASE DIR>/

To run test.py, you will then need to cd into facebook_dcn_traffic and generate the traffic matrix timeseries. This step will generate the traffic matrix snapshots based on facebook's published data center traces in https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p123.pdf. To do so, you will need to do the following:
1) cd <BASE DIR>/facebook_dcn_traffic
2) run: python multiprocessing_trace_parser.py 
3) If you have the protobuf files of the traces already, then you may skip this step (as this step is rather time-consuming). However, you will have to copy the trace files into the proper subdirectories: copy database_aggregationwindow_1.pb into <BASE DIR>/facebook_dcn_traffic/traffic_matrices/clusterA, web_aggregationwindow_1.pb into <BASE DIR>/facebook_dcn_traffic/traffic_matrices/clusterB, hadoop_aggregationwindow_1.pb into <BASE DIR>/facebook_dcn_traffic/traffic_matrices/clusterC, and combined_aggregationwindow_1.pb into <BASE DIR>/facebook_dcn_traffic/traffic_matrices/clustercombined.

Now, with all the dependencies resolved, you can finally run the simulator in test.py. Please also change the directories for finding the facebook dcn traces, as they are currently hard-coded. I will fix this in the future using environment variables.