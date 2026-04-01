from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyAction, MyObservation

class MyEnv(EnvClient[MyAction, MyObservation, State]):
    """Client for the Email Triage Environment."""

    def _step_payload(self, action: MyAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        observation = MyObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
