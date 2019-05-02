# MQLV: Modified Q-Learning for the Vasicek Model

<p align="middle">
  <img src="https://github.com/dagrate/MQLV/blob/master/images/results_plot1.png" width="400"/>
  <img src="https://github.com/dagrate/MQLV/blob/master/images/results_plot2.png" width="390"/>
</p>

MQLV, Modified Q-Learning for the Vasicek Model, is a Python library that proposes a new model-free reinforcement learning approach in the context of financial transactions that follows a mean reverting process. It limits the Q-values over-estimation observed in QLBS, Q-Learner in the Black-Scholes(-Merton) Worlds. Additionally, it extends the simulation to mean reverting stochastic diffusion processes, the Vasicek model. Furthermore, MQLV uses a digital function, or digital option in the financial world, to estimate the future probability of an event, thus widening the scope of the financial application to any other domain involving time series.

Because of the confidentiality of the orginal financial data sets, we propose to use artificially generated data sets, saved in csv files.

----------------------------

## Context

MQLV is part of a research on retail banking transactions where a reinforcement learning approach is used to determine an optimal policy of money management with the aggregated financial transaction of the clients. By using a digital function and a Q-learner, we are able to determine the optimal policy of money management related to retail banking transactions such as credit card transactions or debit card transactions. The objective of the approach is to bring more transparency and more personalized financial advice related to the various requests of the clients of a bank involving loans or credit and debit cards

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

## Citing

If you use the repository, please cite:

To Be Added Soon

----------------------------

## References

```bibtex
@article{halperin2017qlbs,
  title={QLBS: Q-learner in the Black-Scholes (-Merton) worlds},
  author={Halperin, Igor},
  journal={arXiv preprint arXiv:1712.04609},
  year={2017}
}
```
