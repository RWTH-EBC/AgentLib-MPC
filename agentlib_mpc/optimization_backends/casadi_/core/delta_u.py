from agentlib_mpc.models.casadi_model import CasadiParameter
import casadi as ca
import logging

logger = logging.getLogger(__name__)


def get_delta_u_objectives(sys):
    return sys.objective.get_delta_u_objectives()


def get_objective(sys, delta_obj, u_prev, uk, const_par):
    for idx, name in enumerate(sys.controls.ref_names):
        if name == delta_obj.expression.name:
            control_prev = u_prev[idx]
            control_curr = uk[idx]
            delta = control_curr - control_prev
            if hasattr(delta_obj.weight, "sym"):
                weight_value = ca.substitute(delta_obj.weight, sys.model_parameters.full_symbolic, const_par)
            else:
                weight_value = delta_obj.weight
            return weight_value**2 * delta**2
    logger.error(f"Delta U objective {delta_obj.name} could not be matched to any control variable. Returning 0.")

    return 0
