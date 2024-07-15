# agentlib_mpc
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Coverage](https://rwth-ebc.github.io/AgentLib-MPC/main/coverage/badge.svg)](https://rwth-ebc.github.io/AgentLib-MPC/main/coverage/)
[![pylint](https://rwth-ebc.github.io/AgentLib-MPC/main/pylint/pylint.svg)](https://rwth-ebc.github.io/AgentLib-MPC/main/pylint/pylint.html)
[![documentation](https://rwth-ebc.github.io/AgentLib-MPC/main/docs/doc.svg)](https://rwth-ebc.github.io/AgentLib-MPC/main/docs/index.html)

This is a plugin for [AgentLib]https://github.com/RWTH-EBC/AgentLib). 
Includes functions for modeling with [CasADi](https://web.casadi.org/), and using those models in nonlinear MPC, central and distributed (based on ADMM).

See examples and the tutorial in the docs.


## Installation

Install with:

```
pip install agentlib_mpc
```

To install with full dependencies (recommended), run:
```
pip install agentlib_mpc[full]
```



## Optional Dependencies
AgentLib_MPC has a number of optional dependencies:
 
 - **fmu**: Support simulation of FMU models (https://fmi-standard.org/).
 - **ml**: Use machine learning based NARX models for MPC. Currently supports neural networks, gaussian process regression and linear regression. Installs tensorflow, keras and scikit-learn.
 - **interactive**: Utility functions for displaying mpc results in an interactive dashboard. Installs plotly and dash.

Install these like 
````
pip install agentlib_mpc[ml]
````


## Citing AgentLib_MPC

For now, please cite the base framework under https://github.com/RWTH-EBC/AgentLib.

A preprint is available under http://dx.doi.org/10.2139/ssrn.4884846 and can be cited as: 

> Eser, Steffen and Storek, Thomas and Wüllhorst, Fabian and Dähling, Stefan and Gall, Jan and Stoffel, Phillip and Müller, Dirk, A Modular Python Framework for Rapid Development of Advanced Control Algorithms for Energy Systems. Available at SSRN: https://ssrn.com/abstract=4884846 or http://dx.doi.org/10.2139/ssrn.4884846 

When using AgentLib-MPC, please remember to cite other tools that you are using, for example CasADi or IPOPT.

## Acknowledgments

We gratefully acknowledge the financial support by Federal Ministry \\ for Economic Affairs and Climate Action (BMWK), promotional reference 03ET1495A.

<img src="./docs/source/images/BMWK_logo.png" alt="BMWK" width="200"/>