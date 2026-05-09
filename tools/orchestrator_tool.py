from .base_tool import BaseTool


class StateTool(BaseTool):
    name = "state_tool"

    def update_state(self, state, key: str, value):
        setattr(state, key, value)

    def append_history(self, state, event: dict):
        state.history.append(event)
