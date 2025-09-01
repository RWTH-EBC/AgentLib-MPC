# Changelog

## 0.6.7
- #60: Forbid users from creating instance attributes which are also variables in CasadiModel

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
