Alternating Direction Method of Multipliers
-------------------------------------------

In this section, we will learn how to use agentlib for distributed MPC using 
the alternating direction method of multipliers (ADMM). The required example 
files are located in 'examples/admm/'. They include two 
main scripts and two directories with models and config files respectively. 
We simulate the same system as before, however this time the AHU determines 
its mass flow without knowing the system behaviour of the room, creating the 
need for coordination.

### Main script
There are three main scripts. One runs a local version of the ADMM algorithm, which 
operates within a single thread and is suited for simulation and testing. The other 
one runs the agents in separate python processes and communicates through MQTT.
The last one implements a coordinated ADMM, which can be useful, since it helps 
unify parameter setting and provides better convergence criteria.
Here, we will look at the Realtime implementation using multiprocessing.

````python
from agentlib.utils.multi_agent_system import MultiProcessingMAS
import logging
import matplotlib.pyplot as plt
````
The only new import this time is the 
``MultiProcessingMAS`` utility. Unlike the LocalMASAgency we used before, 
the MultiprocessingMAS spawns a separate python process for each agent, 
allowing 
for the true parallelism that would take place in a real-world MAS. However, 
this also requires the condition that simulations are performed in Realtime, 
since time is now the common variable between systems that keeps them in sync.
Now onto the main script.
````python
    env_config = {"rt": True,
                  "strict": True,
                  "factor": 0.1,
                  "t_sample": 60}

    mas = MultiProcessingMAS(agent_configs=['configs\\cooler.json',
                                            'configs\\cooled_room.json',
                                            'configs\\simulator.json'],
                             env=env_config,
                             variable_logging=True)
    mas.run(until=until)
    results = mas.get_results()
````
As explained, we choose a Realtime environment, set it to ``strict`` 
(RuntimeError will be raised if simulation is too slow), and give it a 
``factor`` of 0.1 to speed it up. Finally, we set ``t_sample`` to 60, so we 
will save our results in an interval of 60 seconds. Then, we provide our MAS 
three configs - one for the room controller, one for the AHU controller and 
one to simulate the full system.

### System models
There are three models. The simulation model and the room model are similar to 
the models we used in the MPC examples before, with the main difference 
being in the constraints and cost function. The simulation model omits the 
MPC-related parts of the model, while the room model is the same as before, 
with only the air mass flow term missing from the cost function. The cooler 
model on the other hand is a simple input equals output model of the mass 
flow, including the cost function term that was removed from the room model. 
Therefore, we created a situation, where the room is not explicitly 
penalized for usage of the mass flow anymore, but instead a separate system is.

### Communication via MQTT
For this example, we are not providing the configs in the python script 
itself, but store them separately as json. Both agent configs and configs of 
single modules can be stored in separate json. Let's look at the config file 
``configs/communicators/cooled_room_mqtt.json``.
Since the agents are now using separate processes, we cannot use the 
``local_broadcast`` communicator anymore. Instead, we are using the MQTT 
communicator from the agentlib. The config for an MQTT commincator is a bit 
more complicated than the local_broadcast. After providing 
an id and specifying the type to "mqtt", there are some parameters to provide:\
"url" and "subscriptions". For small test scripts, the url from the snippet 
below will do.

````json
{
  "module_id": "Ag1Com",
  "type": "mqtt",
  "url": "mqtt://test.mosquitto.org",
  "subscriptions": ["Cooler", "Simulation"]
}
````
The subscriptions are a list of agent ids the agent is subscribed to. For more info 
on MQTT topics visit e.g. 
[here](https://www.hivemq.com/blog/mqtt-essentials-part-5-mqtt-topics-best-practices/).
In agentlib, the mqtt communicator sends messages under a topic consisting of 
"/agentlib/<agent_id>". The ``#`` is a wildcard, so by specifying the topics 
in the way above, the agent will receive all messages from the Cooler agent 
and the Simulation agent. The resulting communication structure can be seen 
in the image below:

.. image::  ../images/tutorials/admm_comm.png


### ADMM config
tbd
Let's look at the beginning of the config for the room agent. First of all, 
we see a file path in the list of modules, which points to our communicator 
config. The root of relative filepaths is the directory, where the main 
script is run. 
````json
{
  "id": "CooledRoom",
  "modules": [
            "configs/communicators/cooled_room_mqtt.json",
    {
      "module_id": "admm_module",
      "type": "admm",
      "optimization_backend": {
        "type": "casadi_admm",
        "model": {
          "type": {
            "file": "models/ca_room_model.py",
            "class_name": "CaCooledRoom"
          }
        },
        "solver": {
          "name": "ipopt",
          "options": {
            "print_level": 0
          }
        },
        "results_file": "admm_opt.csv"
      },
````
We can see, that the module type for the controller now reads "admm", and the 
optimization backend type is "casadi_admm". We can also see, that there are 
some new options set for the optimization_backend, namely the solver option. 
The numerical solver name can be chosen from a list of supported solvers 
(currently supported are ``ipopt``, ``sqpmethod``, ``qpoases``). For most purposes,
IPOPT will be the solver of choice. However, we can change the default 
options for the chosen solver. To see applicable options, please refer to 
the documentation of the solver. For IPOPT, an overview of all the options 
can be found [on the official site](https://coin-or.github.io/Ipopt/OPTIONS.html).
In our 
case, we set the print_level to 0 to avoid clutter in the console output.
We also specify a ``results_file``, so we save detailed information about each 
NLP solution in csv format, readable e.g. as a multi-indexed pandas 
Dataframe. 

After providing parameters and inputs in the usual way, let's 
have a look at what changed between the central MPC and the ADMM.

.. note::
    The ``prediction_horizon``, ``time_step`` and ``penalty_factor`` parameters of 
    the ADMM module affect the strucuture of the optimization problem and 
    need to be identical for all modules taking part in the ADMM algorithm. 
    Currently, this is not validated automatically, so care should be taken when
    writing the config. The ``timeout``, ``registration_period`` and 
    ``admm_iter_max`` parameters should also be the same our similar.

````json
      "controls": [
      ],
      "states": [
        {
          "name": "T_0",
          "value": 298.16,
          "ub": 303.15,
          "lb": 288.15
        }
      ],
      "couplings": [
        {
          "name": "mDot_0",
          "alias": "mDotCoolAir",
          "value": 0.05,
          "ub": 0.1,
          "lb": 0}
````
The ``controls`` list is now empty, as the air mass flow is not determined by 
the room anymore. Instead, it is now listed under the new type ``couplings``. 
The couplings are optimization variables, so they should also have upper and 
lower boundaries. ADMM with agentlib is based on consensus, meaning partial 
systems that have to agree on shared variables are optimized. The shared 
variables are identified through their alias. In this 
example, all agents that define a coupling with alias "mDotCoolAir" share 
this variable. The value for the state "T_0" is obtained from the Simulation 
agent, so care should be taken to make sure the alias matches. In this case, 
the default alias of "T_0" will match, since the name exists in the 
simulation model.

Now let's see the config on the side of the cooler:
````json
      "controls": [
        {
          "name": "mDot",
          "value": 0.02,
          "ub": 0.1,
          "lb": 0
        }
      ],
      "states": [
      ],
      "couplings": [
        {
          "name": "mDot_out",
          "alias": "mDotCoolAir",
          "value": 0.05
        }
      ]
    },
````
We can see, that there are two variables of interest, one in ``controls`` and 
one in ``couplings``. The control "mDot" is the actuation that is sent to the 
simulator after optimization. Therefore, the alias of the mass flow in the 
Simulation agent must match "mDot". The coupling "mDot_out" is assigned with 
the alias "mDotCoolAir", which matches the coupling in the room agent.

<details>
    <summary>
    Why do I have to declare two variables, if they mean the same thing?
</summary>
<blockquote>
Because the models follow the FMU standard, where variables are divided 
between inputs, outputs, locals/states and parameters. In this case, our 
cooler model takes a mass flow as an input ("mDot" in this case) and 
produces the same mass flow as an output to other systems ("mDot_out" in 
this case). In a more complex setting, the cooler might have an internal PID 
controller to set the mass flow to its correct value. In that case, "mDot" 
would be setpoint of the mass flow, and "mDot_out" would be the actual mass 
flow.
</blockquote>
</details>
