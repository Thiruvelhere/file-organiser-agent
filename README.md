# File Organiser Agent 🗂️🤖

An intelligent, autonomous file organization agent built with the **Strands Agents SDK**. The agent leverages LLMs to inspect messy directories, categorize files by type or context, create target directories, move files safely, and verify filesystem state upon completion.

---

## 🚀 Features

- **Autonomous Decision Making**: Uses LLMs to inspect directory structures and decide how best to organize files.
- **Strict Verification & Safety**: Verifies that every file move succeeds and confirms resulting directory states.
- **Multiple Provider Support**:
  - **Groq Cloud**: Lightning-fast cloud inference with dynamic model selection and native tool-calling support.
  - **Ollama**: 100% private, local model execution (e.g. `llama3.1`).
  - **xAI Grok**: Compatible with xAI Grok models via OpenAI-compatible endpoints.
- **Windows Optimized**: Path-normalized tools that handle Windows PowerShell and avoid escape/JSON formatting issues.
- **Custom Tools**: Built-in tool calling for:
  - `list_directory`: Lists directory items with PowerShell integration.
  - `find_files`: Locates files matching target extensions.
  - `create_folder`: Safely creates directories recursively.
  - `move_file`: Moves files and verifies their new location on disk.

---

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **OS**: Windows (PowerShell support built-in)
- **API Key** (optional for cloud models):
  - [Groq API Key](https://console.groq.com/) for cloud inference, or
  - [Ollama](https://ollama.com/) running locally for offline inference.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Thiruvelhere/file-organiser-agent.git
   cd file-organiser-agent
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# For Groq cloud inference
GROQ_API_KEY=gsk_your_groq_api_key_here

# (Optional) Specify a custom Groq model:
# GROQ_MODEL=openai/gpt-oss-120b
```

Alternatively, set the environment variable directly in PowerShell:
```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
```

---

## 🏃 Usage

### 1. Running with Groq (`grok_agent.py`)
Run the cloud-powered agent:
```powershell
python grok_agent.py
```
*The agent automatically discovers active models on your Groq account with tool-calling capabilities and begins organizing the configured directory.*

### 2. Running with Ollama (`file_organiser.py`)
Make sure Ollama is running locally:
```powershell
ollama run llama3.1
```
Then execute:
```powershell
python file_organiser.py
```

---

## 📁 Repository Structure

```
├── .gitignore          # Git ignore rules for venv, cache, and secrets
├── requirements.txt    # Python package dependencies
├── file_organiser.py   # Ollama-based local agent
├── grok_agent.py       # Groq/cloud-based agent with dynamic model discovery
└── README.md           # Documentation
```

---

## 🛡️ Safety & Safeguards

- **No Destructive Operations**: The agent has no file deletion or file truncation tools.
- **Pre-Execution Check**: Verifies existence of source files before attempting any move.
- **Post-Execution Verification**: Checks the target directory to verify the file was relocated before proceeding.

---

## 👤 Author

Developed by **[Thiruvel](https://github.com/Thiruvelhere)**
