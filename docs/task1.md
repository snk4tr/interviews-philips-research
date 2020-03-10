# Answers to the questions in the code snippet on quantization

## 1. Model architecture

**Question 1.1: What architecture is that?**  
_Answer_: MobileNetV2 model architecture, with several notable modifications to enable quantization:
- Replacing addition with ``nn.quantized.FloatFunctional``
- Insert ``QuantStub`` and ``DeQuantStub`` at the beginning and end of the network.
- Replace ReLU6 with ReLU
Note: this code is taken from [here](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py>).

**Question 1.2: What task is this model for?**  
_Answer_: This is a network for a classification on 1k classes (ImageNet1k).

## 2. Helper functions

**Question 2.1: What does `accuracy` method do?**  
_Answer_: Computes the accuracy over the `k` top predictions for the specified values of `k`.

## 3. Define dataset and data loaders
**Question 3.1: Will the data be loaded randomly, sequentially or in some other way?**  
_Answer_: For training randomly, for testing sequentially.

**Question 3.2: Will the data loading process be reproducible between different runs of the script?**  
_Answer_: It probably will because of `torch.manual_seed(191009)` in the very top of the code. However, since no other seeds are fixed (numpy, python, cuda) the overall reproducibility is at the risk.

**Question 3.3: What are the sizes of input images? Why are they so?** Discuss resizing and different types of crops here.  
_Answer_: 224 both for train and for test. But in case of test images first get resized to 256 and then a central crop is taken. In case of train, images do not get resized to 256 first and also crops are random. From one hand, it may not be optimzal in terms of distribution match between train and test. From the other hand, there are works that show that you can train on a bit lower res images and then make an inference on bigger ones and it will not drop a performance (known only for classification).

## Post-training static quantization

**Question 4.1: What is fusion of a model?**  
_Answer_: It is a process of merging common layer patterns into a single layer. In this particular case we collapse Conv2D + BatchNorm + ReLu into one layer. It is done for the sake of better performance in terms of both accuracy and speed. Is the interviewee does not now the answer, it is possible to understand from the code for `myModel.fuse_model()`.

## Quantization-aware training

**Question 5.1: What is the data type that train data loader returns?**  
_Answer_: The best possible answer would be `Tuple[torch.Tensor, int]`. Worse - `tuple`.

**Question 5.2: What data type would you use if you had to return not only image+target, but also some meta-info or additional required data.**    
_Answer_: The bese possible answer: data class implemented with `@dataclass` functionality from Python 3.7 and above. A bit worse - `namedtuple` from Python `collections` std lib. A bit worse - `dict` with key - name and value - tensor. The worst - bigger `tuple`.

## Benchmarking quantization results

**Question 6.1: Provide an idea or a code snippet of how the results can be benchmarked.** Ask a candidate to imlement a function for benchmarkig model's performance in terms of speed if it will not take more then 5-7 mins.   
_Answer_: The best possible answer will be an implementation similar to the one below (taken from the original tutorial). It is simple, but not perfect. For instance it uses `time.time()` instead of monotonic time.
```Python
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed
```

**Question 6.2: In case of the code was not implemented or a candidate does not want to imlement it ask general question about implemetnation: what time to use, how to measure it, what is the difference between different types of time (monotonic, wall-clock, CPU, elapsed)?**  
_Answer_: Use monotonic or CPU time depending on what is needed, but the general first choise would be monotonic. Measure without data loading process, save it in some time accomulater that the devide by number of items. Measure several times. You may compute both mean time and std in order to better understand the correctness of benchmarking.
* Monotonic time represents the absolute elapsed wall-clock time since some arbitrary, fixed point in the past. It isn't affected by changes in the system time-of-day clock
* Wall-clock time is the time that a clock on the wall (or a stopwatch in hand) would measure as having elapsed between the start of the process and 'now'
* The user-cpu time and system-cpu time are pretty much as you said - the amount of time spent in user code and the amount of time spent in kernel code
* Elapsed time - the number of seconds that the process has spent on the CPU, including time spent waiting for its turn on the CPU (while other processes get to run).