from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from agentlib_mpc.models.serialized_ml_model import CustomGPR


class GPRTrainer:
    """
    Trains GPR with scikit-learn.
    """

    def __init__(self):
        self.test_gpr = self.build_test_gpr()

    def build_test_gpr(self):
        """
        Builds GPR and returns it.
        """
        kernel = ConstantKernel(constant_value_bounds=(1e-3, 1e5)) * RBF(
            length_scale_bounds=(1e-3, 1e5)
        ) + WhiteKernel(noise_level=1.5, noise_level_bounds=(1e-3, 1e5))

        gpr = CustomGPR(
            kernel=kernel,
            copy_X_train=False,
            n_restarts_optimizer=0,
        )
        return gpr

    def fit_test_gpr(self, data: dict):
        self.test_gpr.fit(
            X=data.get("x"),
            y=data.get("y"),
        )
