from strands import Agent, tool
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import shutil
import time
from collections import defaultdict

load_dotenv()


# ==================================================
# FILESYSTEM TOOLS
# Key idea: tools take/return SUMMARIES, not per-file
# lists, so LLM input/output size never scales with
# the number of files in the directory.
# ==================================================

@tool
def scan_files(path: str) -> str:
    """Scan a directory and return a summary grouped by file extension.

    Only scans files directly inside the specified directory.
    Does not recursively scan subdirectories. Returns extension-level
    counts and total size, NOT a per-file listing, so this stays compact
    even for directories with thousands of files.

    Args:
        path: Directory to scan.
    """
    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Directory does not exist: {path}"

        summary = defaultdict(lambda: {"count": 0, "total_bytes": 0})
        total_files = 0

        for entry in os.scandir(path):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower() or "(no extension)"
                size = entry.stat().st_size
                summary[ext]["count"] += 1
                summary[ext]["total_bytes"] += size
                total_files += 1

        if not summary:
            return "No files found in the directory."

        lines = [f"Total files: {total_files}", f"Unique extensions: {len(summary)}", ""]
        for ext, info in sorted(summary.items(), key=lambda x: -x[1]["count"]):
            mb = info["total_bytes"] / (1024 * 1024)
            lines.append(f"{ext}: {info['count']} files, {mb:.1f} MB total")

        return "\n".join(lines)

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
def organize_by_extension(path: str, mapping: dict[str, str]) -> str:
    """Move every file in the directory into a folder based on its extension.

    This tool internally scans the directory and moves each file according
    to `mapping` (extension -> destination folder name). You only need to
    provide the mapping once, regardless of how many files match each
    extension — the tool handles enumerating and moving all of them.

    Never overwrites an existing file. If the destination already contains
    a file with the same name, that specific file is left in place.
    Any extension not present in `mapping` is left untouched (not moved).

    Args:
        path: Base directory containing the files to organize.
        mapping: Dict mapping file extension (e.g. ".pdf", ".jpg",
            "(no extension)") to a destination folder name (e.g.
            "Documents", "Images"). Destination folders must already exist
            or will be treated as invalid.
    """
    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Base directory does not exist: {path}"

        moved = defaultdict(int)
        skipped_conflict = 0
        skipped_no_folder = 0
        untouched = 0

        for entry in list(os.scandir(path)):
            if not entry.is_file():
                continue

            ext = os.path.splitext(entry.name)[1].lower() or "(no extension)"
            folder_name = mapping.get(ext)

            if folder_name is None:
                untouched += 1
                continue

            destination_folder = os.path.join(path, folder_name)
            if not os.path.isdir(destination_folder):
                skipped_no_folder += 1
                continue

            destination = os.path.join(destination_folder, entry.name)
            if os.path.exists(destination):
                skipped_conflict += 1
                continue

            shutil.move(entry.path, destination)
            moved[folder_name] += 1

        lines = ["Move summary:"]
        for folder_name, count in sorted(moved.items()):
            lines.append(f"  {folder_name}: {count} files moved")
        lines.append(f"Skipped (name conflicts, left in place): {skipped_conflict}")
        lines.append(f"Skipped (destination folder missing): {skipped_no_folder}")
        lines.append(f"Untouched (extension not in mapping): {untouched}")

        return "\n".join(lines)

    except Exception as e:
        return f"Failed to organize files: {e}"


@tool
def verify_directory(path: str) -> str:
    """Verify the final state of a directory.

    Shows each top-level folder and how many files it contains, plus any
    loose files remaining at the top level. Returns folder-level counts,
    NOT a full per-file listing, so this stays compact for large directories.

    Args:
        path: Directory to verify.
    """
    try:
        path = os.path.normpath(path)

        if not os.path.isdir(path):
            return f"Directory does not exist: {path}"

        output = []
        loose_files = 0

        for entry in sorted(os.scandir(path), key=lambda x: x.name.lower()):
            if entry.is_file():
                loose_files += 1
            elif entry.is_dir():
                try:
                    file_count = sum(1 for c in os.scandir(entry.path) if c.is_file())
                except PermissionError:
                    file_count = "?"
                output.append(f"FOLDER: {entry.name} ({file_count} files)")

        if loose_files:
            output.append(f"Loose files remaining at top level: {loose_files}")

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
    raise ValueError("No API key found. Please set GROQ_API_KEY in your environment or .env file.")

base_url = "https://api.groq.com/openai/v1"
model_id = "openai/gpt-oss-120b"

print(f"Using provider at {base_url}")
print(f"Using model: {model_id}")

model = OpenAIModel(
    client_args={"api_key": api_key, "base_url": base_url},
    model_id=model_id
)


# ==================================================
# RETRY WRAPPER FOR RATE LIMITS
# ==================================================

def call_with_retry(fn, *args, max_retries=5, **kwargs):
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
        organize_by_extension,
        verify_directory
    ],
    system_prompt="""
You are an autonomous file organization assistant.

Your job is to organize files inside the directory provided by the user.

STRICT WORKFLOW (follow this exact order, minimizing tool calls):

1. Call scan_files EXACTLY ONCE. It returns extension-level counts,
   not individual filenames.

2. Based on the extensions present, decide a mapping from EACH extension
   to ONE destination folder category. Typical categories: Documents,
   Images, Videos, Audio, Archives, Code, Installers, Other. Do not
   invent categories that aren't useful, and do not create empty folders.

3. Call create_folders EXACTLY ONCE with the full list of category names
   you decided on.

4. Call organize_by_extension EXACTLY ONCE, passing the complete
   extension-to-folder mapping as a single dict argument. This tool moves
   every matching file internally — you do not need to (and should not)
   list individual filenames anywhere.

5. Call verify_directory EXACTLY ONCE to confirm the final state.

6. Report completion in one or two short sentences based on the
   verification result (e.g. "Done. 246 files organized into 8 folders.").
   Do not enumerate individual files in your response.

Never call move_file-style operations one file at a time. Never list
individual filenames in your reasoning or responses — you are working
at the extension/category level only.

OTHER RULES:

- Prefer a simple, shallow folder structure.
- Do NOT recursively reorganize existing subdirectories.
- Do NOT delete any files or modify file contents.
- Do NOT overwrite existing files; conflicts are handled by the tool
  automatically (original file is left in place).
- Only operate inside the directory provided by the user.
- If the directory is already reasonably organized, make few or no changes.

Always format Windows paths using forward slashes in tool calls,
for example: C:/Users/sibir/Downloads
"""
)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    directory = input("Enter the directory to organize: ").strip()
    directory = os.path.abspath(directory)

    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        exit()

    response = call_with_retry(
        agent,
        f"""
Organize this directory: {directory}

Preserve every existing file and its contents. Follow the strict
workflow: scan once, decide the extension-to-folder mapping, create
folders once, organize by extension once, then verify once.
"""
    )

    print(response)