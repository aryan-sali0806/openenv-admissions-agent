# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Admissions Env Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# Import your actual Pydantic models
from models import AdmissionsAction, AdmissionsObservation, AdmissionsState

class AdmissionsEnv(
    EnvClient[AdmissionsAction, AdmissionsObservation, AdmissionsState]
):
    """
    Client for the Admissions Env Environment.
    Handles the seamless translation between Python objects and server JSON.
    """

    def _step_payload(self, action: AdmissionsAction) -> Dict[str, Any]:
        """
        Convert your AdmissionsAction into the JSON the server expects.
        """
        return {
            "action_type": action.action_type,
            "action_args": action.action_args,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdmissionsObservation]:
        """
        Unwrap the server's JSON back into your AdmissionsObservation Pydantic model.
        """
        obs_data = payload.get("observation", {})
        
        # Pydantic magic: **obs_data automatically unpacks the dict into your model!
        observation = AdmissionsObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AdmissionsState:
        """
        Parse the server's internal memory state.
        """
        return AdmissionsState(**payload)