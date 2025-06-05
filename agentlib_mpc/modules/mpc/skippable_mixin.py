from typing import Optional

import agentlib
from agentlib import AgentVariable
from pydantic import Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.data_structures import mpc_datamodels


class SkippableMixinConfig(agentlib.BaseModuleConfig):
    enable_deactivation: bool = Field(
        default=False,
        description="If true, the MPC module uses an AgentVariable `active` which"
        "other modules may change to disable the MPC operation "
        "temporarily",
    )
    deactivation_source: Optional[agentlib.Source] = Field(
        default=None, description="Source for the deactivation signal."
    )
    active: AgentVariable = Field(
        default=AgentVariable(
            name=mpc_datamodels.MPC_FLAG_ACTIVE,
            description="MPC is active",
            type="bool",
            value=True,
            shared=False,
        ),
        validate_default=True,
        description="Variable used to activate or deactivate the MPC operation",
    )

    @field_validator("active")
    def add_deactivation_source(cls, active: AgentVariable, info: FieldValidationInfo):
        source = info.data.get("deactivation_source")
        if source is not None:
            active.source = source
        return active


class SkippableMixin(agentlib.BaseModule):
    config: SkippableMixinConfig

    def check_if_should_be_skipped(self):
        """Checks if mpc steps should be skipped based on external activation flag."""
        if not self.config.enable_deactivation:
            return False
        active = self.get(mpc_datamodels.MPC_FLAG_ACTIVE)

        if active.value == True:
            return False
        source = str(active.source)
        if source == "None_None":
            source = "unknown (not specified in config)"
        self.logger.info("MPC was deactivated by source %s", source)
        return True
