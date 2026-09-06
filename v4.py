import os
import re
import shutil
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ==================================================
# PLAIN PYTHON FILESYSTEM HELPERS (no LLM involved)
# ==================================================

def scan_files(path: str) -> list[dict]:
    """Return metadata for every file directly inside `path`."""
    entries = []
    for entry in os.scandir(path):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower() or "(no extension)"
            entries.append({
                "name": entry.name,
                "extension": ext,
                "size": entry.stat().st_size,
            })
    return entries


def create_folders(base: str, names: list[str]) -> None:
    for name in names:
        folder_path = os.path.join(base, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"CREATED folder: {name}")
        elif not os.path.isdir(folder_path):
            print(f"SKIP folder (file exists with this name): {name}")


def move_files(base: str, moves: list[tuple[str, str]]) -> None:
    """moves: list of (filename, destination_folder_name)"""
    for filename, folder_name in moves:
        source = os.path.join(base, filename)
        destination_folder = os.path.join(base, folder_name)
        destination = os.path.join(destination_folder, filename)

        if not os.path.isfile(source):
            print(f"SKIP (source missing): {filename}")
            continue
        if not os.path.isdir(destination_folder):
            print(f"SKIP (no destination folder): {filename} -> {folder_name}")
            continue
        if os.path.exists(destination):
            print(f"SKIP (name conflict, left in place): {filename}")
            continue

        shutil.move(source, destination)
        print(f"MOVED: {filename} -> {folder_name}")


def verify_directory(path: str) -> str:
    output = []
    for entry in sorted(os.scandir(path), key=lambda x: x.name.lower()):
        if entry.is_file():
            output.append(f"FILE: {entry.name}")
        elif entry.is_dir():
            output.append(f"FOLDER: {entry.name}")
            try:
                for child in sorted(os.scandir(entry.path), key=lambda x: x.name.lower()):
                    if child.is_file():
                        output.append(f"  └── {child.name}")
            except PermissionError:
                output.append("  └── [permission denied]")
    return "\n".join(output) if output else "Directory is empty."


# ==================================================
# GROQ CLIENT (single lightweight classification call)
# ==================================================

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set GROQ_API_KEY in your environment or .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
MODEL_ID = "openai/gpt-oss-120b"


def classify_extensions(extensions: list[str]) -> dict[str, str]:
    """
    Ask the LLM to map each unique file extension to ONE folder category.
    This call is small and cheap regardless of how many files exist,
    because it only ever sees the list of unique extensions, not filenames.
    """

    prompt = f"""
You are classifying file extensions into folder categories for a file
organizer. Categories should typically be chosen from this list, but you
may use a different short category name if none of these fit well:

Documents, Images, Videos, Audio, Archives, Code, Installers, Other

Here are the unique file extensions found in the directory:
{json.dumps(extensions)}

Return ONLY a JSON object mapping each extension to exactly one category
name. No prose, no markdown fences, no explanation. Example format:

{{".pdf": "Documents", ".jpg": "Images", ".exe": "Installers"}}
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them anyway
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw}") from e

    return mapping


# ==================================================
# ORCHESTRATION (plain python, no agent tool-loop)
# ==================================================

def organize_directory(directory: str) -> None:
    files = scan_files(directory)

    if not files:
        print("No files found in the directory. Nothing to organize.")
        return

    unique_extensions = sorted({f["extension"] for f in files})
    print(f"Found {len(files)} files across {len(unique_extensions)} unique extensions.")
    print("Asking model to classify extensions...")

    mapping = classify_extensions(unique_extensions)

    # Fallback: any extension the model missed goes to "Other"
    for ext in unique_extensions:
        if ext not in mapping:
            mapping[ext] = "Other"

    print("\nExtension -> Category mapping:")
    for ext, category in mapping.items():
        print(f"  {ext} -> {category}")

    needed_folders = sorted(set(mapping.values()))
    print(f"\nCreating folders: {needed_folders}")
    create_folders(directory, needed_folders)

    moves = [(f["name"], mapping[f["extension"]]) for f in files]

    print(f"\nMoving {len(moves)} files...")
    move_files(directory, moves)

    print("\nFinal directory state:")
    print(verify_directory(directory))


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    directory = input("Enter the directory to organize: ").strip()
    directory = os.path.abspath(directory)

    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        exit()

    organize_directory(directory)