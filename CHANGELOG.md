# Changelog

## 0.7.0
- Integrate with new mas dashboard from agentlib
- Add state fallback to mpc. Allows the config option "enable_state_fallback" to be set to true (default false). If active, the age of a measurement is checked, and if it is older than the sample time, the predicted state from the last optimization will be used to initialize the problem.
- Fix bugs that occur with newer numpy versions in ADMM.
- Remove relative tolerance from ADMM, must provide tolerance for primal and dual residual in absolute values.

## 0.6.7
- Add option to skip MPC calculation in given time intervals, e.g. during summer period
- Add a fallback pid module that listens to the same deactivation module and switches on

## 0.6.6
- self.time available in mpc and ml mpc (not yet available for admm, minlp, etc)


## 0.6.5
- Fix faulty warning about boundaries of binary controls
- Implement rounding for controls to fix pycombina errors and AgentVariable warnings for minor outliers (tolerance 1e-5)

## 0.6.4
- Fix exchange ADMM

## 0.6.3
- Dashboard now synchronously updates traces of all plots when changing one plot
- Fix some smaller issues with lag structure in data driven mpc
- Add ADMM dashboard


## 0.6.2
- Add moving horizon estimator. Use it as ``"agentlib_mpc.mhe"``


## 0.6.1

- Added Changelog
- Added fatrop support (requires casadi 3.6.6)
- add "euler" to integrator for discretization options in multiple shooting
