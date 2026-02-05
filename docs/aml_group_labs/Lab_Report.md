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

a) We applied the profiling in Task 1 to here, and when the device is set to CPU:
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
b) The purpose of the loop is to take a compressed 8-bit integers (MXINT8) and expand them into 16 bit brain floating point numbers (Bfloat16) so the GPU can perform high-precision math operation. 

However, there might be a problem when the GPU assumes there is a implicit leading bit in front of the mantissa, but MXINT8 is a raw integer without the implicit leading bit. For example: 

Let the MXINT8 value = -5 and shared scale = 2
```
MXINT8 input: 
- Integer (hX) = 10000101 ("1" for the negative sign bit, "0000101" for 5 in binary)
- Shared Scale = exponent bias (127)  + Shared Scale (2) = "10000001" (129 in binary)
```

After bitcast, the code form a 16-bit string: 
```
[Sign: 1][Exp: 10000001][Mantissa: 0001010]*
```
*0001010 is 5 in binary shifted left by 1, this is to align the bits to Bfloat16 binary point. 

However, when GPU interpret this number as a bfloat16, it calculate as: 
```
Sign = 1 -> negative
Exponent = 129 - 127 = 2 -> Exponent of 2
Mantissa =  0.0001010 (fraction) + 1.0000000 (assumed implicit leading bit)= 1.0001010 (1.078125 in decimal)

Hardware final interpretation Output: - 1.078125 * 2^2 = -4.3125

```
In the "dont_need_abs" logic, only the lower 6 bits of MXINT8 integer are put into the fraction and the most significant bit of MXINT8 (the "64" bit) is used to decide whether it is necessary to minus bias to compensate for the "implicit 1". Therefore, the "implicit 1" is designed to represent the value of 64 in Bfloat16. 

Hence, we conclude the dequantisation kernel formula without considering bias are: 
```
(sign)* Bfloat16 = (sign)*(MXINT8 Integer Magnitude/64)*2^(shared_scale)
```
Using this formula, we can check that the corresponding MXINT8 of the previous output are: 
```
 -4.3125/4*64 = -69 -> Magnitude differ by the value of the "implicit 1" = 64
```

Therefore, to solve this problem, the dequantisation kernel formula needs to subtract bias if necessary.
```
y[i] = dont_need_abs ? out : out - bias

If the 7th bit of MXINT8 = 1: # MXINT8 magnitude >= 64 
     = (sign)*(1.0 + MXINT8 Magnitude/64)*2^(shared_scale)

(e.g. 65 = 1000001 in binary -> fraction = 0.000010 when divided by 64 -> after adding the "implicit 1" = 1.0000010 -> 64 + 1 = 65 -> No need for bias reduction because MSB of 1 already act as 64 in 1000001)

Else: # MXINT8 magnitude < 64
     = (sign)*(1.0 + MXINT8 Magnitude/64)*2^(shared_scale) - 
       (1.0*2^(shared_scale))
     = MXINT8 Magnitude/64 * 2^(shared_value)

```
Hence, applying the formula:
```
Corresponding MXINT8 = -0.3125*64/4 = -5
```

To conclude, the variable "dont_need_abs" is a boolean representing the result of a range check that determines if the bias subtraction is necessary. For the variable "bias", it represents the value of the implicit leading bit that bfloat16 automatically added into any bitcasted number. 




