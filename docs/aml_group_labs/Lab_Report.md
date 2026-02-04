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
![Neural Architecture Search Task 1](./attachments/Tutorial_5_task_1.png)
![Neural Architecture Search Task 2](./attachments/Tutorial_5_Task2.png)


# Lab 3: Mixed Precision Search
## Part 1

After running 100 trials with Optuna TPESampler, the maximum achieved accuracy is found to be 0.87216. The plot optimization history plot also shows the best accuracy has been reached in the first 10 trials, with the remaining trials attemping to improve but didn't find any better parameters. 
![Mixed Precision Search](./attachments/T6_1_100epoch.png)

## Part 2
The experiment was setup to run 25 optuna trials, with the TPESamples, where the search space was constricted to a 2 linear layer choices similar to Part 1, but this time with we ran 8 different studies with following supported precisions for the linear layers: 

| Study (precision variant) | Linear layer Search Space | Supported precision values (from Tutorial 6) |
|---|---|---|
| Integer | `torch.nn.Linear` / `LinearInteger` | bit-width \(W\) = [8, 16, 32]; frac-width \(F\) = [2, 4, 8] |
| MinifloatDenorm | `torch.nn.Linear` / `LinearMinifloatDenorm` | total width \(W\) = [8, 16, 32]; exponent width \(E\) = [2, 4, 8]; exponent bias = `None` |
| MinifloatIEEE | `torch.nn.Linear` / `LinearMinifloatIEEE` | total width \(W\) = [8, 16, 32]; exponent width \(E\) = [2, 4, 8]; exponent bias = `None` |
| Log | `torch.nn.Linear` / `LinearLog` | bit-width \(W\) = [8, 16, 32]; exponent bias = [-1, 0, 1] |
| BlockFP | `torch.nn.Linear` / `LinearBlockFP` | width \(W\) = [4, 8, 16]; exponent width \(E\) = [4, 8, 16]; block size = [8, 16, 32] |
| BlockLog | `torch.nn.Linear` / `LinearBlockLog` | bit-width \(W\) = [8, 16, 32]; shared exponent-bias width = [2, 4, 8]; block size = [8, 16, 32] |
| Binary | `torch.nn.Linear` / `LinearBinary` | binary (1-bit) weights; stochastic = [0, 1]; bipolar = `True` |
| BinaryScaling | `torch.nn.Linear` / `LinearBinaryScaling` | binary (1-bit) data/weights/bias; stochastic = [0, 1]; bipolar = `True`; `binary_training` enabled |

The plot below shows the maximum achieved accuracy for each study, with the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. We have multiple curves, showing the maximum achieved performance at different scales.

**Performance Comparison for all suported linear layer precisions**
![NAS Performance Comparison (All)](./attachments/nas_comparison_plot.png)

<table>
  <tr>
    <td align="center" width="50%">
      <b>Figure 2a: High-performing Linear layers (0.86–0.88)</b><br>
      <img src="./attachments/nas_comparison_plot_high.png" width="100%">
    </td>
    <td align="center" width="50%">
      <b>Figure 2b: Mid/low-performing Linear layers (0.50–0.83)</b><br>
      <img src="./attachments/nas_comparison_plot_mid.png" width="100%">
    </td>
  </tr>
</table>

A summary of the best performing linear layers for each precision variant can be found below:

| Precision Type | Best Accuracy | Best Trial ID |
|---|---:|---:|
| BlockFP | 0.87324 | 20 |
| MinifloatIEEE | 0.87228 | 16 |
| Integer | 0.87216 | 4 |
| BlockLog | 0.87196 | 13 |
| MinifloatDenorm | 0.87168 | 21 |
| Log | 0.80764 | 12 |
| Binary | 0.51392 | 19 |
| BinaryScaling | 0.50964 | 16 |

This clearly shows that the BlockFP Layer quantisation performs the best, with the highest accuracy of 0.87324, followed by MinifloatIEEE, and Integer quantisations. The best performing linear layer choices and quantisations for the BlockFP precision (best trial = 20) are:

- **BlockFP config**: `W=4`, `E=4`, `exponent_bias=None`, `block_size=8`
- **Layers changed to BlockFP to**: `L0.query`, `L0.key`, `L0.attn_out_dense`, `L0.ffn_out_dense`, `L1.value`, `L1.ffn_out_dense`, `classifier`
- **Layers retained with FP32**: `L0.value`, `L0.ffn_intermediate`, `L1.query`, `L1.key`, `L1.attn_out_dense`, `L1.ffn_intermediate`, `pooler_dense`

```
Input
  │
Encoder layer 0:
  Q,K = BlockFP; V = FP32
  Attention output dense = BlockFP
  FFN intermediate = FP32; FFN output dense = BlockFP
  │
Encoder layer 1:
  Q,K = FP32; V = BlockFP
  Attention output dense = FP32
  FFN intermediate = FP32; FFN output dense = BlockFP
  │
Pooler dense = FP32
  │
Classifier dense = BlockFP
```

A thing to note about the experiment is that it is limited to trying between only 2 linear layer quantisation choices in every study, and also we were restricted by computuational resources restricting our number of trials to only 25. Running expeirments with more number of trials can lead to finding better performing quantisations for some of the layers, and may even lead to the mid/low performing layers to perform close to our original unquantised baseline model. Similarly running a study with the search space spanning all the possible linear layer quantisation may be computationally expensive but can lead to finding the best performing model.

This part is an effective display that different layers in the Neural network may be quantised with a different precision or precision type to achieve the best performance, given adequate computational resources to perform the hyper-parameter search. 

# Lab 4: (Software Stream) Performance Engineering

### Task 1

a) ```torch.compile``` compiles PyTorch code into optimized kernels that significantly speed up inference. This feature relies on **TorchDynamo** to compile the code into graphs and **TorchInductor** to further compile the graphs into optimized kernels, which is ready for GPU deployment.

When the optimised model is used on CPU, massive overhead will occur due to the limted amount of threading/parallel computing on CPU. The execution will need to allocate memory to save all the data from a thread before the start of the execution of the next thread in CPU. Therefore, it turns out the optimised model rans slower on CPU compared to the original model. 

b) In this task the experiment is ran on Colab, using the T4 GPU. When the device is set to CUDA:
```
Original model: 1.8479 s 
Optimized model: 1.3985 s
```

### Task 2

a) We applied the profiling in Task 1 to here, and when the device is set to CPU, 
```
Unfused SPDA:     546.620 ms
Fused SPDA:       27.650 ms
```
b) When device is set to CUDA (Colab), using the T4 GPU gives the runtime comparison:
```
Unfused SPDA:     0.616 ms
Fused SPDA:       0.257 ms
```
Here no matter CPU or GPU is used, Fused SPDA outperforms the original SPDA implementation. This can be explained as in the fused version, PyTorch implementation reduces the amount of read/write operations by only having a single kernel. It basically removes the need of storing the intermediate results, unlike the normal verison of SPDA where we need 4 kernels to perform a successful SPDA, which requires to record the intermediate products which increased the memory access traffic, which translates to slower completion speed.
### Task 3

