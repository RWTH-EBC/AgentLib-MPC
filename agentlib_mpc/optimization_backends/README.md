The new backend structure looks like this:

````
optimization_backends
 - backend.py
 - casadi_
    - core
        - casadi_backend.py
        - discretization.py
        - system.py
        - VariableGroup.py
    - admm.py
    - basic.py
    - miqp.py
````

The new casadi backend is not one large file and instead is made up from components, for which abstract base classes can be found in the core.

``system.py`` defines the interaction between the model and the optimization and is largely equivalent to what previously was the ``declare_quantities()`` function. \
Much of the old backend is now in ``discretization.py``, where the optimal control problem is created and solved. \
A backend always has a system that is fit for it (and a VariableReference, which defines the boundary between the agentlib and the optimization).
Additionally, it can have multiple discretizations to choose from. Currently collocation is implemented.

The actual implementations of the backend are currently ``admm.py``, ``basic.py``, ``miqp.py``.
They define a system, discretizations and compose the final casadi_backend.
The casadi_backend then has to be added to the ``__init__.py`` to be usable.

I think it would be good for the data-driven mpc to have a separate backend.

