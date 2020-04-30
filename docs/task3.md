# Q&A for the code snippet on the modified CycleGAN model

## 1. Model architecture

- **Q**: Which loss functions are used in this model?  
  **A**: Discriminator loss (BCE/MSE), color loss (to be implemented), lab identity loss (MAE in the lab color space), cycle consistency loss (MSE), GAN loss (BSE/MAE depending on the discriminator loss), feature loss (MSE, something loke Perceptual loss). All these losses are duplicated because CycleGAN has two discriminators and two generators.
- **Q**: What does GAN loss do?  
  **A**: It is either a standard GAN loss (BCE) or MSE on soft labels. `GANLoss` class also creates fake/real target tensors, because it is not possible to compute loss between tensor (model output) and some python data type so it needs to be converted before computations.
- **Q**: Look at `is_train` flag in the `__init__` method. What is its purpose? What else can be added here?  
  **A**: It is made to not load discriminators and feature network (for perceptual loss) on the inference time. We can add one of generators here because we do not need then both on the inference time.


## 2. Code style and structuring 

#### General questions 
- Grade this code from 0 to 10, where 10 is perfect. If the grade is low, tell 3 things that can significantly improve quality of this particular code snippet
- How would you reorganize this code to make it more convenient to understand and use?

#### Specific questions
- Are there any problems with imports?
- Do you like naming of functions/methods/classes? Tell 3 things that you consider to be bad practice and require changes straight away


## 3. Coding

Please implement `color_loss` considering the following requirements (duplicated as comments int the code for candidate's):
1. The loss is computed on pair of image tensors (predict, target) of shapes (n_batches, height, width, n_channels)
2. The loss is computed only on regions of interest (ROIs) that contain object of interest: person or avatar
3. The loss computes RMSE between colors within ROIs of passed tensors.  
Let color be the average value of region of (n x n) pixels.
    
You are allowed to use you favourite search engine to check APIs and docs of used libraries 

#### Reference implementation

```python
def color_loss(predict: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    # Considering corner cases:
    assert predict.shape == target.shape, \
        f'Passed tensors have to have the same shape, got: predict - {predict.shape}, target - {target.shape}'
    assert n > 0, f'Number of pixels for computation of color must be greater then 0, got {n}'
    
    img_shape = input_tn.shape
    grain_sz = 5
    height = 8 * grain_sz
    width = 20 * grain_sz
    top = img_shape[2] - height
    left = (img_shape[3] - width) // 2

    def pooling(tensor):
        n_rows = height // grain_sz
        n_cols = width // grain_sz
        splitted_along_1d = torch.stack(tensor.split(n_rows, dim=2))
        splitted_along_2d = torch.stack(splitted_along_1d.split(n_cols, dim=4))
        loss_shape = tuple([n_cols, n_rows] + list(img_shape[:2]) + [grain_sz * grain_sz])
        return splitted_along_2d.reshape(loss_shape).mean(4)

    pooled_input = pooling(input_tn[:, :, top:top + height, left:left + width])
    pooled_target = pooling(target_tn[:, :, top:top + height, left:left + width])
    mse_loss = get_mse({})
    return mse_loss(pooled_input, pooled_target)
```