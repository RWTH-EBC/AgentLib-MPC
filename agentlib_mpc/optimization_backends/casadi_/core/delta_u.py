import logging
logger = logging.getLogger(__name__)


def get_delta_u_objectives(sys):
    delta_u_objectives = []
    if hasattr(sys, 'objective') and sys.objective is not None:
        try:
            delta_u_objectives = sys.objective.get_delta_u_objectives()
        except (AttributeError, Exception) as e:
            logger.warning(f"Failed to get delta_u_objectives: {str(e)}")
    return delta_u_objectives


def get_objective(sys, delta_obj, u_prev, uk, const_par):
    control_name = delta_obj.get_control_name()
    for idx, name in enumerate(sys.controls.ref_names):
        control_prev = u_prev[idx]
        control_curr = uk[idx]
        delta = control_curr - control_prev

        if hasattr(delta_obj.weight, 'sym'):
            param_found = False
            for i, param_name in enumerate(sys.model_parameters.ref_names):
                if param_name == delta_obj.weight.name:
                    weight_value = const_par[i]
                    param_found = True
                    break

            if not param_found:
                raise ValueError(f"Parameter {delta_obj.weight.name} not found in model parameters")
        else:
            weight_value = delta_obj.weight

        return weight_value ** 2 * delta ** 2

    return 0

