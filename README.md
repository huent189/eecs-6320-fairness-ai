# How to train the model:

```
python main.py --config configs/mlp/vanila.yaml
```
# TODO List:
- Method 1:
    - [x] Write SBS Sampling 
    - [ ] Modify the SBS Sampling 
        - [ ] Read TFRecords in torch
        - [ ] Write a standard Dataloader for dataset that reads embeddings and labels
        - [ ] Write a SBS sampler class or method that returns a SBS sampler
        - [ ] Test the sampler with fake data or st
        - [ ] Integrate into torch-lightning! 
    - [ ] Write code for varying threshold  
- Method 2:
    - [ ] Implement the bias council models  
    - [x] GCE loss function
    - [ ] Hue's new loss function 
- General:
    - [ ] Add image dataset (?)  
- If we have more time:
    - [ ] Ambiguity (?)