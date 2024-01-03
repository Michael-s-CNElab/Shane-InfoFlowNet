# Shane-InfoFlowNet
Discovering causality connectivity between brain regions using attention model

1. Model architecture
![image](figure/Figure1_model.png)
2. Causality inference
![image](figure/Figure2_inference.png)

# Data sets
### 1. simulate data
The code for processing simulate data is in [sim data_process](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/tree/main/sim_data_process>).
### 2. Multitasking EEG data
The code for processing Multitasking EEG data is in [Multitasking_process](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/tree/main/Multitasking_process>).
### 3. Lane-Keeping EEG data
The code for processing Lane-Keeping EEG data is in [LaneKeeping_process](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/tree/main/LaneKeeping_process>).
# Model
InfoFlowNet model code, [without mask](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/tree/main/model/InfoFlowNet>) and [mask](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/tree/main/model/InfoFlowNet_mask>) versions respectively.

Executing the training model:  runModel.py

Executing the inferencing model:  inferenceModel.py

# Other
### 1. TCDF
Please refer to the original thesis:  [Temporal Causal Discovery Framework](<https://github.com/M-Nauta/TCDF?tab=readme-ov-file#prerequisites>)

I modified the [input](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/blob/main/TCDF/processEEGdata.py>) of the TCDF model here.
### 2. GC
Using the [MVGC](<https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgchelp.html>) package, the code is in [Multitasking process](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/blob/main/Multitasking_process/MVGC_timedomain_GC.m>) and [Lane-Keeping process](<https://github.com/Michael-s-CNElab/Shane-InfoFlowNet/blob/main/LaneKeeping_process/MVGC_timedomain_GC.m>).