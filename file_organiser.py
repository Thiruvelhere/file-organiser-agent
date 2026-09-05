from strands import Agent, tool
from strands.models.ollama import OllamaModel
import subprocess
import os
import shutil


@tool
def list_directory(path: str) -> str:
    """List files and folders in a Windows directory."""

    result = subprocess.run(
        [
            "powershell",
            "-Command",
            f"Get-ChildItem -LiteralPath '{path}'"
        ],
        capture_output=True,
        text=True
    )

    if result.stdout:
        return result.stdout

    if result.stderr:
        return f"Error: {result.stderr}"

    return "Directory is empty."


@tool
def find_files(path: str, extension: str) -> str:
    """Find files with a specific extension in a directory.

    Args:
        path: Directory to search.
        extension: File extension to look for, such as .txt or .py.
    """

    try:
        files = []

        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)

            if (
                os.path.isfile(full_path)
                and filename.lower().endswith(extension.lower())
            ):
                files.append(full_path)

        if not files:
            return f"No {extension} files found."

        return "\n".join(files)

    except Exception as e:
        return f"Failed to find files: {e}"


@tool
def create_folder(path: str) -> str:
    """Create a folder at the specified path."""

    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created folder: {path}"

    except Exception as e:
        return f"Failed to create folder: {e}"


@tool
def move_file(source: str, destination_folder: str) -> str:
    """Move a file into a destination folder.

    Args:
        source: Full path of the actual file to move.
        destination_folder: Full path of the destination folder.
    """

    try:
        if not os.path.isfile(source):
            return f"Source file does not exist: {source}"

        os.makedirs(destination_folder, exist_ok=True)

        filename = os.path.basename(source)

        destination = os.path.join(
            destination_folder,
            filename
        )

        shutil.move(source, destination)

        return f"Successfully moved {filename} to {destination_folder}"

    except Exception as e:
        return f"Failed to move file: {e}"


model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.1"
)


agent = Agent(
    model=model,
    tools=[
        list_directory,
        find_files,
        create_folder,
        move_file
    ]
)


response = agent("""
You are a file organization agent.

You may ONLY modify:

C:\\Users\\sibir\\Desktop\\agent_test\\messy

Your task is to move all .py files into a folder called:

C:\\Users\\sibir\\Desktop\\agent_test\\messy\\pyfiles

Follow these steps:

1. create the pyfiles folder.
2. find all the .py files in the messy directory.
3. move each .py file found in the messy directory to the newly created pyfiles folder

4. Do not delete or modify any files.

After completing the moves, use list_directory to verify the result.
""")

print(response)