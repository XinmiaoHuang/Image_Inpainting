# Implementation of neural network for Image inpainting
## Environment
* python 3.7.7
* pytorch 1.4.0

## Test
1. run generate_mask.py for mask generation
````
    python generate_mask.py
````
2. run Test_model.py for testing
````
    python Test_model.py
````
Data path in options.py need to be modified before testing.

## Result
On coco2014 Test images:

![Image1](https://github.com/XinmiaoHuang/Image_Inpainting/blob/master/pic/image.png)

There are problems of artifacts and color difference for some inferior or failure cases:

![Image2](https://github.com/XinmiaoHuang/Image_Inpainting/blob/master/pic/inferior1.png)


