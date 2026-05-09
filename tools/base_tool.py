class BaseTool:
    name: str = "base_tool"
    version: str = "1.0"

    def health_check(self) -> dict:
        return {
            "tool": self.name,
            "version": self.version,
            "status": "ok",
        }
