from strands import Agent, tool
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import shutil
import time

load_dotenv()


# ==================================================
# FILESYSTEM TOOLS (BATCHED)
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
def create_folders(path: str, names: list[str]) -> str:
    """Create one or more folders inside a base directory in a single call.

    Skips folders that already exist. Never overwrites files.

    Args:
        path: Base directory in which to create the folders.
        names: List of folder names to create (e.g. ["Documents", "Images"]).
    """

    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Base directory does not exist: {path}"

        results = []

        for name in names:
            folder_path = os.path.normpath(os.path.join(path, name))

            if os.path.exists(folder_path):
                if os.path.isdir(folder_path):
                    results.append(f"EXISTS: {name}")
                else:
                    results.append(f"SKIP (file with this name exists): {name}")
                continue

            os.makedirs(folder_path)
            results.append(f"CREATED: {name}")

        return "\n".join(results)

    except Exception as e:
        return f"Failed to create folders: {e}"


@tool
def move_files(moves: list[dict]) -> str:
    """Move multiple files into destination folders in a single call.

    Never overwrites an existing file. If the destination already contains
    a file with the same name, that specific move is skipped and the
    original file is left in place.

    Args:
        moves: List of move instructions, each a dict with:
            - "source": full path of the file to move
            - "destination_folder": full path of the destination folder
    """

    try:
        results = []

        for move in moves:
            source = os.path.normpath(move.get("source", ""))
            destination_folder = os.path.normpath(move.get("destination_folder", ""))

            if not os.path.isfile(source):
                results.append(f"SKIP (source missing): {source}")
                continue

            if not os.path.isdir(destination_folder):
                results.append(f"SKIP (destination folder missing): {destination_folder}")
                continue

            filename = os.path.basename(source)
            destination = os.path.join(destination_folder, filename)

            # Safety: never overwrite an existing file
            if os.path.exists(destination):
                results.append(
                    f"SKIP (name conflict, left in place): {filename}"
                )
                continue

            shutil.move(source, destination)

            if os.path.isfile(destination):
                results.append(f"MOVED: {filename} -> {destination_folder}")
            else:
                results.append(f"FAILED (not found at destination): {filename}")

        return "\n".join(results)

    except Exception as e:
        return f"Failed to move files: {e}"


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
# RETRY WRAPPER FOR RATE LIMITS
# ==================================================

def call_with_retry(fn, *args, max_retries=5, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise


# ==================================================
# STRANDS AGENT
# ==================================================

agent = Agent(
    model=model,
    tools=[
        scan_files,
        create_folders,
        move_files,
        verify_directory
    ],
    system_prompt="""
You are an autonomous file organization assistant.

Your job is to organize files inside the directory provided by the user.

STRICT WORKFLOW (follow this exact order, minimizing tool calls):

1. Call scan_files EXACTLY ONCE to inspect the directory.

2. Decide ALL the folder categories you will need, in one pass.
   Then call create_folders EXACTLY ONCE with the full list of folder
   names for that base directory. Do NOT call create_folders multiple
   times.

3. Decide the destination for EVERY file that needs to move, in one pass.
   Then call move_files EXACTLY ONCE with the complete list of moves.
   Do NOT call move_files multiple times and do NOT move files one at a time.

4. Call verify_directory EXACTLY ONCE to confirm the final state.

5. Report completion based on the verification result.

Do not call scan_files, create_folders, move_files, or verify_directory
more than once each, except in the rare case a call fails and must be
retried. Never call move_files in a loop, one file per call — always
batch every move into a single call.

CATEGORIES:

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
   Do not create speculative or empty folders.

OTHER RULES:

- Prefer a simple, shallow folder structure.
- Do NOT recursively reorganize existing subdirectories.
  Treat existing folders as potentially intentional.
- Do NOT move files that are already appropriately organized.
- Do NOT delete any files.
- Do NOT modify file contents.
- Do NOT overwrite existing files. If a destination already contains a
  file with the same name, leave the original file where it is.
- Only operate inside the directory provided by the user.
  Never access unrelated directories.
- The goal is useful organization, NOT maximum reorganization.
  If the directory is already reasonably organized, make few or no changes.

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


response = call_with_retry(
    agent,
    f"""
Organize this directory:

{directory}

Preserve every existing file and its contents.

You have authority to create useful folders and move files
within this directory.

Follow the strict workflow: scan once, create all folders in one
call, move all files in one call, then verify once.
"""
)

print(response)