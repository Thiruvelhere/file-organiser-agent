
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import subprocess
import os
import shutil

load_dotenv()



@tool
def list_directory(path: str) -> str:
    """List files and folders in a Windows directory."""

    path = os.path.normpath(path)
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
        path = os.path.normpath(path)
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
        path = os.path.normpath(path)
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
        source = os.path.normpath(source)
        destination_folder = os.path.normpath(destination_folder)

        if not os.path.isfile(source):
            return f"Source file does not exist: {source}"

        os.makedirs(destination_folder, exist_ok=True)

        filename = os.path.basename(source)

        destination = os.path.join(
            destination_folder,
            filename
        )

        shutil.move(source, destination)

        # Verify that the move actually happened
        if os.path.isfile(destination):
            return f"Move verified: {filename} is now in {destination_folder}"

        return f"Move failed: {filename} was not found at the destination."

    except Exception as e:
        return f"Failed to move file: {e}"


# --------------------------------------------------
# Groq / Grok model configuration
# --------------------------------------------------

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "No API key found. Please set GROQ_API_KEY in your environment or in a .env file."
    )

base_url = "https://api.groq.com/openai/v1"
model_id = "openai/gpt-oss-120b"

print(f"Using provider at {base_url} with model: {model_id}")

model = OpenAIModel(
    client_args={
        "api_key": api_key,
        "base_url": base_url
    },
    model_id=model_id
)


model = OpenAIModel(
    client_args={
        "api_key": api_key,
        "base_url": base_url
    },
    model_id=model_id
)


# --------------------------------------------------
# Strands Agent
# --------------------------------------------------

agent = Agent(
    model=model,
    tools=[
        list_directory,
        find_files,
        create_folder,
        move_file
    ],
    system_prompt=(
        "You are an expert file organization assistant. "
        "Always format all file paths in tool call arguments using forward slashes "
        "(e.g., 'C:/Users/sibir/Desktop/agent_test/messy') rather than backslashes to avoid JSON formatting errors."
    )
)


# --------------------------------------------------
# Run the agent
# --------------------------------------------------
directory = input("Enter the directory to clean: ").strip()
directory = os.path.abspath(directory)

if not os.path.isdir(directory):
    print(f"Directory does not exist: {directory}")
    exit()
response = agent(f"""
Take responsibility for cleaning up:

{directory}

Inspect the workspace first and improve its organization so that a developer could easily understand where everything belongs.

Preserve all existing files and their contents. Do not delete anything.

You have authority to create directories and move files within this workspace when appropriate.

Use your judgment to determine the organization.

When you believe you're finished, verify the resulting filesystem state rather than relying on assumptions.

Only report completion after verification.
""")

print(response)

