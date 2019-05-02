# MQLV: Modified Q-Learning for the Vasicek Model

<p align="middle">
  <img src="https://github.com/dagrate/MQLV/blob/master/images/results_plot1.png" width="400"/>
  <img src="https://github.com/dagrate/MQLV/blob/master/images/results_plot2.png" width="390"/>
</p>

MQLV is a Python and R library that proposes to evaluate the quality of the generated adversarial samples using persistent homology. For some real-world applications, different than computer vision, we cannot assess visually the quality of the generated adversarial samples. Therefore, we have to use other metrics. Here, we rely on persistent homology because it is capable to acknowledge the shape of the data points, by opposition to traditional distance measures such the Euclidean distance.

The generative models are trained with Python to produce adversarial samples saved in csv files.

The persistent homology features and the bottleneck distance are evaluated with the TDA package of R. 


----------------------------

## Dependencies

The library uses **Python 3** with the following modules:
- numpy (Python 3)
- scipy (Python 3)
- matplotlib (Python 3)
- pandas (Python 3)
- bspline (Python 3)

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

----------------------------

%## Citing

%If you use the repository, please cite:
