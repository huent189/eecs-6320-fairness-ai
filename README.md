# Datasets:
- Link to Mimic-CXR-Embedding dataset in numpy formart: [google drive](https://drive.google.com/file/d/1q-dLbuhitcwJsrLKR09OSW6WOzIv4soi/view?usp=drive_link)

# How to train the model:

```
python main.py --config configs/mlp/vanila.yaml
```
# TODO List:
- Method 1:
    - [x] Write SBS Sampling 
    - [x] Modify the SBS Sampling 
        - [x] Read TFRecords in torch
        - [x] Write a standard Dataloader for dataset that reads embeddings and labels
        - [x] Write a SBS sampler class or method that returns a SBS sampler
        - [ ] Test the sampler with fake data or st
        - [x] Integrate into torch-lightning! 
    - [x] Write code for varying threshold  
        - [x] test the OptimalThreshold module
        - [x] Metrics:
            - [x] Add the proposed formula instead of f1 score
            - [x] Write a new Optimal Threshold selector (It should find the best threshold for each group)
        - [x] Dataset:
            - [x] Write a dataset for the method
        - [x] Trainers:
            - [x] Write a new MIMICTrainer (pass demographic info to the threshold selector)
                - [x] Should acquire thershold using validation or train set after training is finished
                - [x] Should measure discripencies after test is finished
        - [ ] Give test dataloadrs name so they show up with proper names in the final log files
        - [ ] Check with alex whether the formula should be FPR or FNR
    - [ ] Run experiments
- Method 2:
    - [ ] Implement the bias council models  
    - [x] GCE loss function
    - [ ] Hue's new loss function 
- General:
    - [x] Add dataset with labels for each disease (vs. No Finding/Finding). 
    - [ ] Add image dataset (?)  
    - [ ] Add other metrics:
        - [ ] FPR 
        - [ ] AUC 
- Prof Notes:
     - [ ] AUC & FPR both before and after 
     - There should be some disparity before and after applying debiasing the bias should be decreased for all groups and they should be more aligned. 
     - After doing debiasing the AUC should not drop largely. 
     - AUC on average of all classes and on No finding
- If we have more time:
    - [ ] Ambiguity (?)

- Logging stuff needed:
 - [x] Exact thresholds found for each group 
 - [x] Full FPR metrics 
 - [ ] FDR rate
 - [x] Log accucary for baseline and SBS
 - [x] Log everything in the threshold scenario