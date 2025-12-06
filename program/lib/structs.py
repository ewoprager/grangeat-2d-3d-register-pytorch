class Error:
    def __init__(self, description: str):
        self.description = description

    def __str__(self) -> str:
        return f"{self.description}"

    def __repr__(self) -> str:
        return f"Error(description='{self.description}')"
