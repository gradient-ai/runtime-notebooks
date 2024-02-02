# Accelerate Demo

Use `nlp_example.py` script to test. 

### Pre-req:
```
pip install evaluate
```

### Run on Single GPU:
`python ./nlp_example.py ` -> Watch usage on `nvidia-smi`

### Run on Multi GPU:
- Install pre-req
- Setup Accelerate config (if using for the first time)
```
accelerate config 


In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0

Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2

How many different machines will you use (use more than 1 for multi-node training)? [1]: 1

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no

Do you wish to optimize your script with torch dynamo?[yes/NO]    :no

Do you want to use DeepSpeed? [yes/NO]: no

Do you want to use FullyShardedDataParallel? [yes/NO]: no

Do you want to use Megatron-LM ? [yes/NO]: no

How many GPU(s) should be used for distributed training? [1]: 2

Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: no
```

- launch `accelerate launch ./nlp_example.py` 
- watch `nvidia-smi` -> should work as expected!