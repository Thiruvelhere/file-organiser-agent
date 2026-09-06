from strands import Agent, tool
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import shutil

load_dotenv()


# ==================================================
# FILESYSTEM TOOLS
# ==================================================

@tool
def scan_files(path: str) -> str:
    """Scan a directory and return a compact list of files.

    Only scans files directly inside the specified directory.
    Does not recursively scan subdirectories.

    Args:
        path: Directory to scan.
    """

    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Directory does not exist: {path}"

        files = []

        for entry in os.scandir(path):
            if entry.is_file():
                size = entry.stat().st_size
                extension = os.path.splitext(entry.name)[1].lower()

                files.append(
                    f"{entry.name} | type={extension or 'no extension'} | size={size} bytes"
                )

        if not files:
            return "No files found in the directory."

        return "\n".join(sorted(files))

    except Exception as e:
        return f"Failed to scan directory: {e}"


@tool
def create_folder(path: str) -> str:
    """Create a folder at the specified path.

    Args:
        path: Full path of the folder to create.
    """

    try:
        path = os.path.normpath(path)

        if os.path.exists(path):
            if os.path.isdir(path):
                return f"Folder already exists: {path}"
            return f"Cannot create folder because a file exists at: {path}"

        os.makedirs(path)

        return f"Successfully created folder: {path}"

    except Exception as e:
        return f"Failed to create folder: {e}"


@tool
def move_file(source: str, destination_folder: str) -> str:
    """Move a file into a destination folder.

    Never overwrites an existing file.

    Args:
        source: Full path of the file to move.
        destination_folder: Full path of the destination folder.
    """

    try:
        source = os.path.normpath(source)
        destination_folder = os.path.normpath(destination_folder)

        if not os.path.isfile(source):
            return f"Source file does not exist: {source}"

        if not os.path.isdir(destination_folder):
            return f"Destination folder does not exist: {destination_folder}"

        filename = os.path.basename(source)
        destination = os.path.join(
            destination_folder,
            filename
        )

        # Safety: never overwrite an existing file
        if os.path.exists(destination):
            return (
                f"Move skipped: destination already contains "
                f"a file named '{filename}'."
            )

        shutil.move(source, destination)

        # Verify move
        if os.path.isfile(destination):
            return (
                f"Move verified: '{filename}' "
                f"is now in '{destination_folder}'."
            )

        return f"Move failed: '{filename}' was not found at destination."

    except Exception as e:
        return f"Failed to move file: {e}"


@tool
def verify_directory(path: str) -> str:
    """Verify the final state of a directory.

    Shows folders and the files contained directly within them.

    Args:
        path: Directory to verify.
    """

    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Directory does not exist: {path}"

        output = []

        for entry in sorted(os.scandir(path), key=lambda x: x.name.lower()):

            if entry.is_file():
                output.append(f"FILE: {entry.name}")

            elif entry.is_dir():
                output.append(f"FOLDER: {entry.name}")

                try:
                    for child in sorted(
                        os.scandir(entry.path),
                        key=lambda x: x.name.lower()
                    ):
                        if child.is_file():
                            output.append(
                                f"  └── {child.name}"
                            )
                except PermissionError:
                    output.append("  └── [permission denied]")

        if not output:
            return "Directory is empty."

        return "\n".join(output)

    except Exception as e:
        return f"Failed to verify directory: {e}"


# ==================================================
# GROQ MODEL
# ==================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "No API key found. Please set GROQ_API_KEY "
        "in your environment or in a .env file."
    )

base_url = "https://api.groq.com/openai/v1"
model_id = "openai/gpt-oss-120b"

print(f"Using provider at {base_url}")
print(f"Using model: {model_id}")


model = OpenAIModel(
    client_args={
        "api_key": api_key,
        "base_url": base_url
    },
    model_id=model_id
)


# ==================================================
# STRANDS AGENT
# ==================================================

agent = Agent(
    model=model,
    tools=[
        scan_files,
        create_folder,
        move_file,
        verify_directory
    ],
    system_prompt="""
You are an autonomous file organization assistant.

Your job is to organize files inside the directory provided by the user.

CORE RULES:

1. First inspect the directory using scan_files.

2. Organize files based on their type and purpose.

3. Use your judgment when deciding categories.
   Typical categories may include:
   - Documents
   - Images
   - Videos
   - Audio
   - Archives
   - Code
   - Installers
   - Other

   Do not create categories unless they are actually useful.

4. Prefer a simple, shallow folder structure.

5. Do NOT recursively reorganize existing subdirectories.
   Treat existing folders as potentially intentional.

6. Do NOT create speculative folders.

7. Do NOT create empty folders.

8. Do NOT move files that are already appropriately organized.

9. Do NOT delete any files.

10. Do NOT modify file contents.

11. Do NOT overwrite existing files.

12. If a destination already contains a file with the same name,
    leave the original file where it is.

13. Minimize filesystem operations.
    Do not move a file multiple times.

14. Only operate inside the directory provided by the user.
    Never access unrelated directories.

15. After making changes, use verify_directory to inspect the
    resulting filesystem.

16. Only report completion after verification.

IMPORTANT:

The goal is useful organization, NOT maximum reorganization.

If the directory is already reasonably organized,
make few or no changes.

Always format Windows paths using forward slashes in tool calls,
for example:

C:/Users/sibir/Downloads
"""
)


# ==================================================
# RUN
# ==================================================

directory = input(
    "Enter the directory to organize: "
).strip()

directory = os.path.abspath(directory)

if not os.path.isdir(directory):
    print(f"Directory does not exist: {directory}")
    exit()


response = agent(f"""
Organize this directory:

{directory}

Preserve every existing file and its contents.

You have authority to create useful folders and move files
within this directory.

Inspect first, organize intelligently, minimize unnecessary
changes, and verify the final filesystem state.
""")

print(response)