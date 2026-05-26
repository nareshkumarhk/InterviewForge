from pathlib import Path
from typing import Dict


class PromptManager:
    """Manages prompt versions and loading."""

    def __init__(self, version: str = "v1.0", base_path: str = "prompts"):
        self.version = version
        self.base_path = Path(base_path)
        self.prompts_path = self.base_path / version

        if not self.prompts_path.exists():
            raise ValueError(f"Prompt version {version} not found at {self.prompts_path}")

    def load(self, prompt_name: str) -> str:
        """
        Load a prompt by name.

        Args:
            prompt_name: Name without extension (e.g., "evidence_extraction")

        Returns:
            Prompt content as string
        """
        prompt_file = self.prompts_path / f"{prompt_name}.txt"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()

    def load_all(self) -> Dict[str, str]:
        """
        Load all prompts in the version directory.

        Returns:
            Dictionary of {prompt_name: content}
        """
        prompts = {}
        for prompt_file in self.prompts_path.glob("*.txt"):
            prompt_name = prompt_file.stem
            prompts[prompt_name] = self.load(prompt_name)
        return prompts

    def get_version(self) -> str:
        """Get current prompt version."""
        return self.version
