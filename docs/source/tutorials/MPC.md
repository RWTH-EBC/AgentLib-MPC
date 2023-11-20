Model Predictive Control
------------------------

### Creating an MPC in agentlib with CasADi
To run a model predictive controller, a system model for use in optimization 
is required. What model types are available depends on the chosen 
optimization backend. In this section, creating an MPC with a CasADi backend 
is explained. \
Open the 'examples/multi-agent-systems/casadi/simple_mpc.py' example.
#### Imports
As usual, let's look at the imports first.
```python
import logging
from typing import List
import matplotlib.pyplot as plt
from agentlib.models.casadi_model import CasadiModel, CasadiInput, CasadiState, \
    CasadiParameter, CasadiOutput
from agentlib.utils.multi_agent_system import LocalMASAgency
```
We import logging as usual and 
``typing`` is used to annotate the optimzation model we will be creating, 
and matplotlib is used to plot the results. Next, we import the CasadiModel and 
some CasadiVariables. We will use these to specify an agentlib-style 
CasadiModel. Finally, we import the LocalMASAgency utility. This can be used 
to conveniently create and run your local MAS, without creating the agents 
and their environment by hand.

#### Model creation

Now let's see how we can create an optimization model. The model contains 
the physical system dynamics, as well as the cost function and additional 
constraints on the system.

.. note:: 
    In future versions of agentlib, cost function and constraints will be 
    separated from the model and instead specified in the module config.

In this example, we will create a model of a room, which is under a constant 
heat load and can be controlled by changing the mass flow of cool air from 
an air handling unit.

.. image::  ../images/tutorials/room_example.png

Creating a custom CasadiModel is similar to creating a module. Three steps 
are required:
1. Creating a class that inherits from ``CasadiModel`` 
2. Declare the model variables in the inner class ``Settings``
   - inputs
   - outputs
   - states
   - parameters 
3. Define model equations by overwriting the ``setup_system`` method

##### Variable declaration

Let's see, how we declare the variables required for our simple room model. 
Since modeling in agentlib is based on the FMU-standard, we divide our 
variables into inputs, outputs, parameters and locals (called states to 
avoid clash with the python builtin _locals_). First, we need to create a custom 
config for our CasadiModel. 
````python  
class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(name="mDot", value=0.0225, unit="K", description="Air mass flow into zone"),

        # disturbances
        CasadiInput(name="load", value=150, unit="W", description="Heat "
                                                                  "load into zone"),
        CasadiInput(name="T_in", value=290.15, unit="K", description="Inflow air temperature"),

        # settings
        CasadiInput(name="T_upper", value=294.15, unit="K", description="Upper boundary (soft) for T."),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(name="T", value=293.15, unit="K", description="Temperature of zone"),

        # algebraic

        # slack variables
        CasadiState(name="T_slack", value=0, unit="K", description="Slack variable of temperature of zone")
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(name="cp", value=1000, unit="J/kg*K", description="thermal capacity of the air"),
        CasadiParameter(name="C", value=100000, unit="J/K",
                        description="thermal capacity of zone"),
        CasadiParameter(name="s_T", value=1, unit="-", description="Weight for T in constraint function"),
        CasadiParameter(name="r_mDot", value=1, unit="-",
                        description="Weight for mDot in objective function")
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name='T_out', unit="K", description="Temperature of zone")
    ]
````
Our room model has four inputs. These include the inputs of the physical 
system, being the air mass flow from the AHU, the temperature of this mass flow and 
the load on the system. We also count the upper room temperature limit as an 
input, since it should be settable by the occupants of the room. \
To declare an input, we put a CasadiInput object into a list _inputs_. A 
variable always needs a name. You can also give it a value, which will be 
used if no other value is provided at Runtime. The _unit_ and _description_ 
parameters currently serve no purpose, but can be helpful to readers of the 
model. \
Next we define the states. For one, that is the temperature of the room. 
Since we use soft constraints to enforce an adequate room temperature, we also have to include 
a slack variable. 

.. note:: 
    States in the context of an AgentLib model refers to all variables that 
    are local to a model. All differential variables have to be declared as 
    states, but not all states need to be associated with a 
    differential equation.

Next, we have the parameters. These include the specific thermal capacity of 
air, the thermal capacity of the room and two weights for the cost function. 
Finally, we specify an output of the model. It is not required for the MPC 
in this example, but can be useful for situations, where one might want to 
use the same model for optimization and simulation. Outputs always need to be 
associated with an algebraic equation.

##### Equation and constraints
After specifying a config, we can write the model class itself, which containts the 
dynamics. First, it is important to specify the ``config_type`` attribute of the 
class and set it to the config class we defined. \
The model equations and constraints are specified in the ``setup_system`` method.
We can access the variables defined above by referencing `self.<name>`. 
Differential equations are associated with a variable by setting the ``ode`` 
attribute of that variable. In the same way, algebraic equations can be 
defined by setting the ``alg`` attribute.

````python
class MyCasadiModel(CasadiModel):

    config:MyCasadiModelConfig

    def setup_system(self):
        # Define ode
        self.T.ode = self.cp * self.mDot / self.C * \
                       (self.T_in - self.T) + \
                       self.load / self.C

        # Define ae
        self.T_out.alg = self.T 

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [

            # soft constraints
            (0, self.T + self.T_slack, self.T_upper),

        ]

        # Objective function
        objective = sum([
            self.r_mDot * self.mDot,
            self.s_T * self.T_slack ** 2,
        ])

        return objective
````
Constraints can be added to the model through the ``constraints`` attribute. 
It should be defined as a list of tuples, with the lower bound coming first, 
the constraint function coming second and the upper bound coming last. 
Equality constraints can be added by setting upper and lower bound to the 
same value. Note that algebraic equations will also be converted to equality 
constraints internally. Here, we set one constraint to implement the soft 
constraint on the room temperature.

<details>
    <summary>
    What's the difference between an algebraic equation and setting an 
equality constraint?
</summary>
<blockquote>
Algebraic equations are explicit assignments to a CasadiState or a 
CasadiOutput. They are considered when simulating the model or when doing 
MPC with it. 
Constraints specified as tuples can be of implicit nature, however they are 
ignored for simulation. The only limitation on constraints is, that 
variables that make up the upper or lower bound cannot be used as 
optimization variables in the MPC.
</blockquote>
</details>

.. note::
    Python intuition tells us ``self.<name>`` should not work, as we did not 
    set the attribute.
    In the model base class of agentlib, the ``__get_attr__`` method is written 
    in a way that allows access to all variables that are defined in the 
    Config class of the model.

Finally, we can specify and return the objective function in the same way as 
the other equations. We use the ``sum()`` function from python to 
improve readability.

#### Configuration of the multi-agent-system
Let's look at the environment config first. 
````python
ENV_CONFIG = {"rt": False,
              "t_sample": 60}
````
This time, we specify 'rt' 
(=Realtime) as False, meaning we want the simulation to run as fast as possible.
The 't_sample' option specifies the time step in which the interal clock of 
the environment ticks. This is relevant e.g. for classical controllers like 
PID. It will also affect the sampling with which results are saved.

Below is the config for the MPC agent. As before, we specify an "id" and a 
list of modules, with the first one being a local_broadcast communicator.
Then, we add the MPC module. We specify "mpc" as the type, and then add the 
other options. A central part of the MPC is its _optimization_backend_. The 
optimization backend is specified by another dictionary, always consisting 
of "type" and "model". The model will usually be user-specified and usually 
is provided with the same syntax of "file" and "class_name" as the custom 
module in the PingPong example. The optimization backend also takes an 
option "discretization_options", however we will look at that later.

````python
AGENT_MPC = {"id": "myMPCAgent",
             "modules": [
                 {"module_id": "Ag1Com",
                  "type": "local_broadcast"},
                 {"module_id": "myMPC",
                  "type": "mpc",
                  "optimization_backend":
                      {"type": "casadi",
                       "model": {
                           "type": {"file": __file__,
                                    "class_name": "MyCasadiModel"}},
                       ... 
                       },
                  "time_step": 900,
                  "prediction_horizon": 5,
                  "parameters": [
                      {"name": "s_T", "value": 3},
                      {"name": "r_mDot", "value": 1},
                        ],
                  "inputs": [
                      {"name": "load", "value": 150},
                      {"name": "T_upper", "value": ub},
                      {"name": "T_in", "value": 290.15},
                        ],
                  "controls": [{"name": "mDot", "value": 0.02, "ub": 1, "lb": 0}],
                  "states": [{"name": "T", "value": 298.16, "ub": 303.15, "lb": 288.15}],
                  },
                 ]}
                 ]}
````
Aside from that, "time_step" and "prediction_horizon" need to be specified.
The other options the MPC module takes are ``parameters``, ``inputs``, ``controls`` and 
``states`` . The time step should be provided 
in seconds. The states in the MPC config refer to differential variables, 
not to be confused with states in the model, which refer to any internal 
variables. \
Quantities declared in the module config are variables of the multi-agent-system and 
can be shared with other modules of the same agent, and communicated with 
other agents. All agent variables declared here must match - in name - their 
counterpart in the provided model. Controls, states and inputs must be 
provided fully matching the model. Outputs can be ignored if they are not 
required. Finally, parameters can be omitted, if a default value is provided 
in the model definition. Here, the weight parameters in the cost function 
are provided, as it might be required to change them. However, physical 
parameters such as the thermal capacity of air are taken from the model, as 
they are not expected to change. \
A variable is given a name and a value. For states, the value will determine 
the initial value of the differential variable, if it is not provided 
externally, for example by a simulation agent. Since controls and 
states are the variables of the optimization 
problem, boundaries should be provided via the keys "ub" and "lb". These 
values are for constant hard boundaries. If time-variant boundaries are 
required, one should declare an additional variable and constraint in the model.

#### Running the multi-agent-system

Now that we have our control agent setup, we need to simulate our system. 
The easiest way to do this in agentlib, is to setup an agent with a 
``simulator`` module. Usually in agentlib, we would use an FMU to simulate a 
system. In this example, we will use the CasadiModel we created for the 
optimization. The resulting agent config is shown below.

````python
AGENT_SIM = {"id": "SimAgent",
             "modules": [
                 {"module_id": "Ag1Com",
                  "type": "local_broadcast"},
                 {"module_id": "room",
                  "type": "simulator",
                  "model": {"type": {"file": __file__,
                                     "class_name": "MyCasadiModel"},
                            "states": [
                                {"name": "T", "value": 298.16}
                            ]},
                  "t_sample": 60,
                  "outputs": [
                      {"name": "T_out", "alias": "T"},
                  ],
                  "inputs": [
                      {"name": "mDot", "value": 0.02, "alias": "mDot"},
                  ]},
             ]}
````

The model type for the simulator is provided in the same manner as before. 
However, here we can see, that we have the option to provide additional 
variable options to the model. For example, here we change the starting 
value of the temperature to a value above the upper (soft) boundary, so our 
controller has to work.

Then, inputs and outputs of the simulator. Every 
simulator needs to be provided with a sampling rate "t_sample" in seconds. 
Additionally, declare the output "T_out". This is the first time we use the 
_alias_ keyword. The alias is part of the duo of ``alias`` and ``source`` that uniquely 
define a variable within the MAS. The source is the combination of the ``agent_id`` 
and the ``module_id`` where the variable was defined. When expecting variables from 
another agent, only the ``agent_id`` has to be specified, and when the variable is from 
a module within the same agent, the module_id shoudld be specified. The 
``alias`` is a name independent of the variable name in models (think of long Modelica names) 
and is consistent across agents for the same variable. By default, the name 
of a variable is also its alias. In the case of T_out however, we have to 
specify that this is the variable we want to send to the MPC for its initial 
state. Since the state in the MPC agent is named "T" (and by default has 
alias "T"), we have to set "T" as the alias for our model output. \
The model in turn takes the computed mass flow setpoint as input, hence we 
also have to declare the input. Since the names mDot already match between 
simulator and MPC, the explicit alias declaration is redundant in this case. 
Additionally, we set a default value for "mDot", which is used before the 
first value is received from the MPC.

<details>
    <summary>
    Why are some values specified in the model and others in the module?
</summary>
<blockquote>
Before every step, the simulator gets the current input values from the 
agent and sets them to the model. After performing the step, the outputs 
from the model are written to the agent. Since states per definition are 
internal to the model, they are not set by the agent and their initial 
values have to be changed in the model itself. The same goes for parameters.
</blockquote>
</details>

With all of the setup done, we can now see our MAS run.
Last time, we manually created the agents and the environment. This time, we 
use the ``LocalMASAgency`` utility to setup the system and save results. By 
setting the ``variable_logging`` option to True, time series of all agent 
variables present in the system will be saved. After running the MAS, we can 
retrieve and plot the results of our simulation.
````python
def run_example(with_plots=True):
    mas = LocalMASAgency(agent_configs=[AGENT_MPC, AGENT_SIM],
                         env=ENV_CONFIG,
                         variable_logging=True)
    mas.run(until=10000)
    results = mas.get_results()
````