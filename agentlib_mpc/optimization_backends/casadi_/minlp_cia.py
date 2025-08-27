import os
from typing import Dict

import casadi as ca
import numpy as np
import pydantic
from agentlib.core.errors import OptionalDependencyError
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import (
    MPCVariable,
    MINLPVariableReference,
    stats_path,
    cia_relaxed_results_path,
)
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import (
    CasadiBackendConfig,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import Results
from agentlib_mpc.optimization_backends.casadi_.minlp import CasADiMINLPBackend

try:
    import pycombina
except ImportError:
    raise OptionalDependencyError(
        used_object="Pycombina",
        dependency_install=".\ after cloning pycombina. Instructions: "
        "https://pycombina.readthedocs.io/en/latest/install.html#",
    )


class CasadiCIABackendConfig(CasadiBackendConfig):
    @pydantic.field_validator("overwrite_result_file")
    @classmethod
    def check_overwrite(cls, overwrite_result_file: bool, info: FieldValidationInfo):
        """Checks, whether the overwrite results sttings are valid, and deletes
        existing result files if applicable."""
        res_file = info.data.get("results_file")
        relaxed_res_file = cia_relaxed_results_path(res_file)
        if res_file and info.data["save_results"]:
            if overwrite_result_file:
                try:
                    os.remove(res_file)
                    os.remove(mpc_datamodels.stats_path(res_file))
                    os.remove(relaxed_res_file)
                    os.remove(mpc_datamodels.stats_path(relaxed_res_file))
                except FileNotFoundError:
                    pass
            else:
                if os.path.isfile(info.data["results_file"]):
                    raise FileExistsError(
                        f"Results file {res_file} already exists and will not be "
                        f"overwritten automatically. Set 'overwrite_result_file' to "
                        f"True to enable automatic overwrite it."
                    )
        return overwrite_result_file


class CasADiCIABackend(CasADiMINLPBackend):
    """
    Class doing optimization with the CIA decomposition algorithm.
    """

    # system_type = CasadiMINLPSystem
    # discretization_types = {DiscretizationMethod.collocation: DirectCollocation}
    # system: CasadiMINLPSystem
    var_ref: MINLPVariableReference
    config_type = CasadiCIABackendConfig

    def __init__(self, config: dict):
        super().__init__(config)
        self._created_rel_file: bool = False  # flag if we checked the rel file location

    def solve(self, now: float, current_vars: dict[str, MPCVariable]) -> Results:
        # collect and format inputs
        mpc_inputs = self._get_current_mpc_inputs(agent_variables=current_vars, now=now)

        # solve NLP with relaxed binaries
        relaxed_results = self.discretization.solve(mpc_inputs)

        relaxed_binary_array = self.make_binary_array(full_results=relaxed_results)
        binary_array = self.do_pycombina(b_rel=relaxed_binary_array)

        mpc_inputs_new = self.constrain_binary_inputs(
            mpc_inputs_old=mpc_inputs,
            binary_array=binary_array,
        )
        # solve NLP with fixed binaries
        full_results_final = self.discretization.solve(mpc_inputs_new)

        self.save_rel_result_df(relaxed_results, now=now)
        self.save_result_df(full_results_final, now=now)

        return full_results_final

    def make_binary_array(self, full_results: Results):
        """
        get the binary control variables for input of pycombina and their control vector indexes
        """

        b_rel = [full_results[var] for var in self.var_ref.binary_controls]
        b_rel_np = np.vstack(b_rel)

        # clip binary values within tolerance
        tolerance = 1e-5
        bin_array = b_rel_np
        bin_array = np.where(
            (-tolerance < bin_array) & (bin_array < 0),
            0,
            np.where((1 < bin_array) & (bin_array < 1 + tolerance), 1, bin_array),
        )

        # add additional row to fulfill pycombinas Special Ordered Sets of
        # type 1 condition
        if len(bin_array) == 1:
            ones = np.full(bin_array.shape[1], 1, dtype=float)
            diff = ones - np.sum(bin_array, axis=0)
            diff[diff < 0] = 0
            bin_array = np.vstack([bin_array, diff])

        return bin_array

    def do_pycombina(self, b_rel):
        # N = self.discretization.options.prediction_horizon
        # dt = self.discretization.options.time_step
        # time_end = N * dt
        grid = self.discretization.grid(self.system.binary_controls).copy()
        grid.append(grid[-1] + self.config.discretization_options.time_step)
        # grid = np.linspace(0, time_end, N + 1)
        binapprox = pycombina.BinApprox(
            t=grid,
            b_rel=b_rel,
        )
        # binapprox.set_n_max_switches([3, 1, 0])
        # binapprox.set_min_up_times([3600, 1800, 0])

        bnb = pycombina.CombinaBnB(binapprox)
        bnb.solve(
            use_warm_start=False,
            max_cpu_time=15,
            verbosity=0,
        )
        b_bin = binapprox.b_bin

        # if there is only one mode, we created a dummy mode which we remove now
        if len(self.var_ref.binary_controls) == 1:
            b_bin = b_bin[0, :].reshape(1, -1)

        return b_bin

    def constrain_binary_inputs(
        self,
        mpc_inputs_old: Dict[str, ca.DM],
        binary_array: np.ndarray,
    ) -> dict[str, ca.DM]:
        """

        Args:
            mpc_inputs_old:
            binary_array:

        Returns:

        """

        mpc_inputs_new = mpc_inputs_old.copy()
        name = self.system.binary_controls.name
        mpc_inputs_new[f"lb_{name}"] = binary_array
        mpc_inputs_new[f"ub_{name}"] = binary_array
        return mpc_inputs_new

    def save_rel_result_df(
        self,
        results: Results,
        now: float = 0,
    ):
        """
        Save the results of `solve` for relaxed MINLP into a dataframe at each time step.

        Example results dataframe:

        value_type               variable              ...     lower
        variable                      T_0   T_0_slack  ... T_0_slack mDot_0
        time_step                                      ...
        2         0.000000     298.160000         NaN  ...       NaN    NaN
                  101.431499   297.540944 -149.465942  ...      -inf    0.0
                  450.000000   295.779780 -147.704779  ...      -inf    0.0
                  798.568501   294.720770 -146.645769  ...      -inf    0.0
        Args:
            results:
            now:

        Returns:

        """
        res_file = cia_relaxed_results_path(self.config.results_file)
        if not self.rel_results_file_exists():
            results.write_columns(res_file)
            results.write_stats_columns(stats_path(res_file))

        df = results.df
        df.index = list(map(lambda x: str((now, x)), df.index))
        df.to_csv(res_file, mode="a", header=False)

        with open(stats_path(res_file), "a") as f:
            f.writelines(results.stats_line(str(now)))

    def rel_results_file_exists(self) -> bool:
        """Checks if the relaxed results file already exists, and if not, creates it with
        headers."""
        if self._created_rel_file:
            return True

        res_file = cia_relaxed_results_path(self.config.results_file)

        if res_file.is_file():
            # todo, this case is weird, as it is the mistake-append
            self._created_rel_file = True
            return True

        # we only check the file location once to save system calls
        res_file.parent.mkdir(parents=True, exist_ok=True)
        self._created_rel_file = True
        return False
