# Getting Started: AgentLib-MPC

This is a getting-started tutorial for the AgentLib **AgentLib-MPC** plug-in.

Before starting this tutorial, it helps to complete the AgentLib getting-started tutorial from the main AgentLib repository.

## Tutorial Structure

This tutorial is split into three parts:

1. Setting up a basic multi-agent system using the AgentLib-MPC MPC module
2. Defining objective functions
3. Advanced configuration settings for the MPC module

For every part, you will find a dedicated main file and a dedicated mpc module within the mpc folder. The main file hold all basic instructions for their respective part of the tutorial.

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

where 60 is just an arbitrary time and can be set to whatever. If you are using AgentLib 0.8.9 or newer, apply this change before proceeding with the tutorial.