# Q&A for the "Shoot in you leg" code snippet

## Find mistakes in the provided script

_Answer_:
- [ ] The first import statement is incorrect. One cannot import two modules with a single `as` statement
- [ ] Unused import `import torch.nn.functional as F`
- [ ] There may be no reason to implement your own `Flatten` module since there is one in the PyTorch library (`torch.flatten`)
- [ ] If you still decide to implement `Flatten` then at least minimal `__init__` with a call of `super()` is required
- [ ] Two many output channels in the first `nn.Conv2d` layer. Do not expect good results from such architecture
- [ ] First block of convolutions does not have any non-linearities, which means that it can be reduces to just one conv block
- [ ] First max pooling is too big. Too much information can be lost
- [ ] There is no correspondence between number of channels between the first block of three convolutions and the second one: 512 output channel in the first and 6 input channels in the second
- [ ] Kernel sizes in the second block of 3 convolutions is too big (20, 20)
- [ ] Softmax is not used as activation function in CNNs in backbone layers. In this particular case it is also implicit since it is not clear what dimention will be chosed for reduction (`dim=0`?). However, it does not break the code. For example the following snippet will return something and will not crash:
    ```python
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3)),
        nn.Softmax())

    tensor = torch.rand((3, 3, 10, 10))
    ```
- [ ] Shapes in the `Linear` layer are incorrect
- [ ] Dropout should not be applied after the last activation layer
- [ ] The last pair of `Sigmoid` - `Softmax` are confused. They should be placed in the reverse order