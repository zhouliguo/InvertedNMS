# GRR Face Detection
Image Grid Recognition and Regression Fast and Accurate Face Detection

This repository is under building

C++ Version: https://github.com/zhouliguo/FaceDetection

## Download

Trained Models: https://drive.google.com/drive/folders/1niPITB5tU4aC-NDy4mkAmzePkYeKfXxV?usp=sharing

## Test
### Evaluate WIDER FACE
1. cd GRR
2. python eval.py --weights='weight path' --source='WIDER FACE path'+'/WIDER_val/images/' --save-path='save path'

### Detect Demo
1. cd GRR
2. python detect.py --weights='weight path'  --image-path='image path'

## Train
1. code the labels by codelabel.py
2. python train.py --data='data path'

## Comparison of Accuracy

### WIDER FACE
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/wider.png">

### DarkFace, DFD and MAFA
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/dtm.png">

## Comparison of Speed

### Light Model on Intel i7-5930K CPU
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/light.png">

## Detection examples

### WIDER FACE
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/wider_example.png">

### DARK FACE
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/dark_example.png">
