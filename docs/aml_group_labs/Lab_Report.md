### Authors: Jun Tan, Souparna Mandal, Clarence Ong, Tanson Tang


# Lab 1: Model Compression (Quantization and Pruning)

## Task 1
| Quantisation Level (Fraction width) | PTQ Accuracy | QAT Accuracy |
|:--------------------:|:-------------:|:--------------:|
| 4 (2)                 | 0.50000      | 0.50000       |
| 8  (4)                | 0.79808      | 0.83936       |
| 12 (6)                | 0.83492      | 0.84160       |
| 16 (8)               | 0.83472      | 0.84192       |
| 20 (10)               | 0.83580      | 0.84192       |
| 24 (12)                | 0.83556      | 0.84192       |
| 28 (14)                | 0.83556      | 0.84176       |
| 32 (16)                | 0.83564      | 0.84192       |


![PTQ vs QAT Comparison](./attachments/QAT.png)

We identified that with 12-bit quantisation (6 bit of fraction width length) as the elbow point of the graph, which gives the best PTQ accuracy out of all runs, and reached identical performance on QAT accuracy compared to higher fixed-point width runs.

## Task 2
Based on Part 1, we decided to use the model with quantisation level of 12 with 6 bits fraction bit length for this task.

| Sparsity level |   L1-norm   |  Random   |
|:---------------:|:------------:|:---------:|
| 0.1            |      0.85868      |     0.84072     |
| 0.2            |      0.85768      |     0.82544     |
| 0.3            |      0.85368     |     0.80696     |
| 0.4            |      0.84652      |     0.78704     |
| 0.5            |      0.84084      |     0.73360     |
| 0.6            |      0.82732      |     0.52348    |
| 0.7            |      0.80168      |     0.50464     |
| 0.8            |      0.74236      |     0.50068     |
| 0.9            |      0.52600      |     0.50084     |


![Pruning Sparsity with methods](./attachments/Pruning_results.png)

The best Pruning result in terms of evaluation accuracy for both ```L1-Norm``` and ```Random``` methods happened when sparsity is set to 0.1. As sparsity value increases, the model's evaluation accuracy reduces, particularly for ```L1-Norm``` method sharp drop occured between 0.8-0.9, while for ```Random``` method sharp drop happended eariler between 0.5-0.6. This can be explained as more weights and connections are dropped in high sparsity values, the BERT model will not be able to learn the representation of the network effectively.
# Lab 2: Neural Architecture Search
![Neural Architecture Search Task 1](.\attachments\Tutorial_5_task_1.png)
![Neural Architecture Search Task 2](.\attachments\Tutorial_5_Task2.png)


# Lab 3: Mixed Precision Search
## Part 1

After running 100 trials with Optuna TPESampler, the maximum achieved accuracy is found to be 0.87216. The plot optimization history plot also shows the best accuracy has been reached in the first 10 trials, with the remaining trials attemping to improve but didn't find any better parameters. 
![Mixed Precision Search](.\attachments\T6_1_100epoch.png)

## Part 2


# Lab 4: (Software Stream) Performance Engineering

### Task 1

a) ```torch.compile``` compiles PyTorch code into optimized kernels that significantly speed up inference. This feature relies on **TorchDynamo** to compile the code into graphs and **TorchInductor** to further compile the graphs into optimized kernels, which is ready for GPU deployment.

When the optimised model is used on CPU, massive overhead will occur due to the limted amount of threading/parallel computing on CPU. The execution will need to allocate memory to save all the data from a thread before the start of the execution of the next thread in CPU. Therefore, it turns out the optimised model rans slower on CPU compared to the original model. 

b) In this task the experiment is ran on Colab, using the A100 GPU. When the device is set to CUDA:
```
Original model: 1.8479 s 
Optimized model: 1.3985 s
```