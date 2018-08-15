# Python implementation of Super Resolution by Simple Function
Implementation of Fast Direct Super-Resolution by Simple Functions [Fast Direct Super-Resolution by Simple Functions](http://faculty.ucmerced.edu/mhyang/papers/iccv13_superresolution.pdf)

An easy way to produce high resolution images using conventional methods including dictionary.

## Dependencies
numpy, opencv2, sklearn

## Usage
You should first add Training data into the 'Train' folder and add Set14 into 'Test'.  
At the begining you should create your model by training them. 
`python kmeans.py`

then you should calculate the coef matrix
`python calculateCoef.py`

final you can test the result
`python compare.py`
**you should add your own image to test it.**
