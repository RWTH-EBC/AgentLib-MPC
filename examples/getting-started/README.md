# Getting Started: AgentLib-MPC

This is a getting-started tutorial for the AgentLib **AgentLib-MPC** plug-in.

Before starting this tutorial, it helps to complete the AgentLib getting-started tutorial from the main AgentLib repository:
https://github.com/RWTH-EBC/AgentLib/tree/main/examples/getting-started

## Tutorial Structure

This tutorial is split into three parts:

1. Setting up a basic multi-agent system using the AgentLib-MPC MPC module
2. Defining objective functions
3. Advanced configuration settings for the MPC module

For every part, you will find a dedicated main file and a dedicated MPC module within the `mpc` folder. The main file holds all basic instructions for their respective part of the tutorial.

In part 2, after the static plots are shown, an interactive dashboard is opened. It displays the MPC results and, if available, the solver statistics. This is a helpful tool to further test out how the different objective terms affect the MPC behavior.

## General MPC Settings

1. In the configurations you need to define the time step and the prediction horizon of an MPC
   - `time_step`: Time step after which the optimization problem is solved again
   - `prediction_horizon`: The prediction horizon over which the MPC looks into the future and over which the optimization is performed
2. Discretization: The MPC optimization is discretized either with
   - Multiple Shooting ```"discretization_options": {
                    "method": "multiple_shooting"
                }```
   - Collocation ```"discretization_options": {
                    "collocation_order": 2,
                    "collocation_method": "radau"
                }```
     - Possible Collocation Methods: radau or legendre
- Further information to discretization and solver settings can be found in `mpc/part3_config.py`

## Notes

### AgentLib >= 0.8.9: sampling configuration change

AgentLib version 0.8.9 introduced a slightly altered way to define the time sampling rate.

In the simulator configuration, instead of writing

```json
{
    "t_sample": 60
}
```

you need to write

```json
{
	"t_sample_simulation": 60,
	"t_sample_communication": 60
}
```

where `60` is just an example value and can be adjusted according to the specific use case. If you are using AgentLib 0.8.9 or newer, apply this change before proceeding with the tutorial.