import os
from pathlib import Path
from livekit.agents import function_tool

@function_tool
def read_file(path: str) -> str:
    """Read the content of a file."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Overwrites existing content."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@function_tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory."""
    try:
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"

@function_tool
def create_directory(path: str) -> str:
    """Create a new directory."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory {path}"
    except Exception as e:
        return f"Error creating directory: {e}"
