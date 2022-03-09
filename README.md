# InvertedNMS
Inverted Non-maximum Suppression for more Accurate and Neater Face Detection

## Download

Trained Models: https://drive.google.com/file/d/1kR7cSvU2Wbhu4QrUcay-iZCBaYVXWbbV/view?usp=sharing

## Test

### Detect Demo
1. cd InvertedNMS
2. python detect.py --weights='weight path'  --image-path='image path'

### Evaluate WIDER FACE
1. cd InvertedNMS
2. python eval.py --weights='weight path' --source='WIDER FACE path'+'/WIDER_val/images/' --save-path='save path'

## Comparison of Accuracy

### WIDER FACE
<img src="https://github.com/zhouliguo/GRR/blob/main/figures/wider.png">
