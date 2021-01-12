# NPLIE
This code implements the paper, "Illumination Estimation for Nature Preserving Low Light Image Enhancement". This implementation may differ from the description in the paper, because the description is ambiguously. If you find this code is helpfully and have futher advice, please feel free to contact me.

## Difference
For computing the weight matrix *****"G"**, the first-order derivative in horizonal and vertical direction are combined together,then logarithmic is implemented on it instead of in both direction separately.

## Usage
Simply,run the command "python main.py -i 'dataPath'" in your console. The 'dataPath' specifies the testing image directory.


## Sample Results
input image|our result
----|-----
![4ori](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/4.bmp)|![4](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/4.jpg)
![6ori](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/6.bmp)|![6](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/6.jpg)
![7ori](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/7.bmp)|![7](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/7.jpg)
![9ori](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/9.bmp)|![9](https://github.com/DavidQiuChao/NPLIE/blob/main/figs/9.jpg)
