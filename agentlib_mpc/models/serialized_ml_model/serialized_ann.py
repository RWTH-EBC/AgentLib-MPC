from copy import deepcopy
from typing import Union, Optional

import numpy as np
from agentlib.core.errors import OptionalDependencyError

try:
    from keras import Sequential
except ImportError as err:
    raise OptionalDependencyError(
        dependency_install="keras",
        used_object="Neural Networks",
    ) from err
from pydantic import Field, ConfigDict

from agentlib_mpc.data_structures.ml_model_datatypes import Feature, OutputFeature
from agentlib_mpc.models.serialized_ml_model.serialized_ml_model import (
    SerializedMLModel,
    MLModels,
)


class SerializedANN(SerializedMLModel):
    """
    Contains Keras ANN in serialized form and offers functions to transform
    Keras Sequential ANNs to SerializedANN objects (from_ANN) and vice versa (deserialize).

    attributes:
        structure: architecture/structure of ANN saved as json string.
        weights: weights and biases of all layers saved as lists of np.ndarrays.
    """

    weights: list[list] = Field(
        default=None,
        title="weights",
        description="The weights of the ANN.",
    )
    structure: str = Field(
        default=None,
        title="structure",
        description="The structure of the ANN as json string.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: MLModels = MLModels.ANN

    @classmethod
    def serialize(
        cls,
        model: "Sequential",
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """Serializes Keras Sequential ANN and returns SerializedANN object"""
        structure = model.to_json()
        weights = []
        for layer in model.layers:
            weight_l = layer.get_weights()
            for idx in range(len(weight_l)):
                weight_l[idx] = weight_l[idx].tolist()
            weights.append(weight_l)

        return cls(
            structure=structure,
            weights=weights,
            dt=dt,
            input=input,
            output=output,
            trainer_config=training_info,
        )

    def deserialize(self) -> "Sequential":
        """Deserializes SerializedANN object and returns a Keras Sequential ANN."""
        from keras import models

        ann = models.model_from_json(self.structure)
        layer_weights = []
        for layer in self.weights:
            l_weight = []
            layer_weights.append(l_weight)
            for matrix in layer:
                l_weight.append(np.asarray(matrix))

        for i, layer in enumerate(ann.layers):
            layer.set_weights(layer_weights[i])
        return ann

    def to_dict(self) -> dict:
        """Transforms self to a dictionary and the numpy arrays to lists, so they can
        be serialized."""
        ann_dict = deepcopy(self.__dict__)
        for layer in ann_dict["weights"]:
            for idx in range(0, len(layer)):
                layer[idx] = layer[idx].tolist()
        return ann_dict
