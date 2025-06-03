from keras.losses import Loss
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PINNLoss(Loss):
    def __init__(self, main_loss_function, additional_losses, n_outputs, weight_phys_losses, **kwargs):
        super().__init__(**kwargs)
        self.main_loss_function = main_loss_function
        self.additional_losses = additional_losses
        self.n_outputs = n_outputs
        self.weight_phys_losses = weight_phys_losses

    def call(self, y_true, y_pred):
        import tensorflow as tf
        # Main loss function
        main_loss_func = tf.keras.losses.get(self.main_loss_function)
        loss_data = main_loss_func(y_true, y_pred)

        # Add physical losses
        loss_phys = 0
        for add_loss in self.additional_losses:
            y_phy = add_loss['function'](y_true)
            loss = main_loss_func(y_phy, y_pred)
            loss_phys += loss / add_loss['scale']

        total_loss = loss_data*(1-self.weight_phys_losses) + loss_phys * self.weight_phys_losses

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "main_loss_function": self.main_loss_function,
            "additional_losses": self.additional_losses,
            "n_outputs": self.n_outputs,
            "weight_phys_losses": self.weight_phys_losses
        })
        return config




