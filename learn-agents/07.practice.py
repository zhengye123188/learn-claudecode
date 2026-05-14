import json
import os
import subprocess
from pathlib import Path
import re
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
TASKS_DIR = WORKDIR / ".tasks"
SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."
# -- Base tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

class TaskManager:
    def __init__(self, task_dir: Path):
        self.dir = task_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self)->int:
        """
            Scan TASKS_DIR = WORKDIR / ".tasks" for all task_*.json files
            and return the maximum ID.

            Args:
                WORKDIR: Path object or string representing the working directory

            Returns:
                int: Maximum task ID found, or 0 if no task files exist
            """
        # Check if the directory exists
        if not self.dir.exists() or not self.dir.is_dir():
            return 0
        # Pattern to match task_*.json files
        pattern = re.compile(r'^task_(\d+)\.json$')
        max_id = 0
        # Iterate through all files in the tasks directory
        for file_path in self.dir.iterdir():
            if file_path.is_file():
                match = pattern.match(file_path.name)
                if match:
                    task_id = int(match.group(1))
                    max_id = max(max_id, task_id)
        return max_id

    def _load_task(self, task_id: int) -> dict:
        """
            Load the task JSON file for a given task ID.

            Args:
                WORKDIR: Path object or string representing the working directory
                task_id: Integer or string representing the task ID

            Returns:
                dict: The loaded JSON content as a dictionary

            Raises:
                FileNotFoundError: If the tasks directory doesn't exist or the task file doesn't exist
                json.JSONDecodeError: If the JSON file is malformed
            """

        # Construct the task file path
        task_file = self.dir / f"task_{task_id}.json"
        # Check if the file exists
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        # Load and return the JSON content
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f) # 返回字典

    # 将dict转换为json文件
    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create_task(self,subject:str,description:str="")->str:
        task = {
            "id":self._next_id,
            "subject": subject,
            "description": description,
            "status":"pending",
            "blockedBy":[],
            "owner":"",
        }
        self._save(task)
        self._next_id += 1
        # 直接将新创建的任务返回给大模型看
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get_by_task_id(self, task_id: int) -> str:
        return json.dumps(self._load_task(task_id), indent=2, ensure_ascii=False)

    def update_task(self,task_id:int,status:str=None,
                    add_blocked_by:list=None,remove_blocked_by:list=None)->str:
        task = self._load_task(task_id)
        if status:
            if status not in ["pending", "completed","in_progress"]:
                raise ValueError(f"Task status {status} is not valid")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by)) # 能去重
        if remove_blocked_by:
            task["blockedBy"] = [a for a in task["blockedBy"] if a not in remove_blocked_by]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, complete_task_id:int):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if task["id"] == complete_task_id:
                continue
            if complete_task_id in task["blockedBy"]:
                task["blockedBy"].remove(complete_task_id)
                self._save(task)

    def list_tasks(self)->str:
        tasks = []
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)

TASKS = TaskManager(TASKS_DIR)
TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create_task(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update_task(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list":   lambda **kw: TASKS.list_tasks(),
    "task_get":    lambda **kw: TASKS.get_by_task_id(kw["task_id"]),
}
TOOLS = [
  {
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string"
        }
      },
      "required": ["command"]
    }
  },
  {
    "name": "read_file",
    "description": "Read file contents.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "limit": {
          "type": "integer"
        }
      },
      "required": ["path"]
    }
  },
  {
    "name": "write_file",
    "description": "Write content to file.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "content": {
          "type": "string"
        }
      },
      "required": ["path", "content"]
    }
  },
  {
    "name": "edit_file",
    "description": "Replace exact text in file.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "old_text": {
          "type": "string"
        },
        "new_text": {
          "type": "string"
        }
      },
      "required": ["path", "old_text", "new_text"]
    }
  },
  {
    "name": "task_create",
    "description": "Create a new task.",
    "input_schema": {
      "type": "object",
      "properties": {
        "subject": {
          "type": "string"
        },
        "description": {
          "type": "string"
        }
      },
      "required": ["subject"]
    }
  },
  {
    "name": "task_update",
    "description": "Update a task's status or dependencies.",
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "integer"
        },
        "status": {
          "type": "string",
          "enum": ["pending", "in_progress", "completed"]
        },
        "addBlockedBy": {
          "type": "array",
          "items": {
            "type": "integer"
          }
        },
        "removeBlockedBy": {
          "type": "array",
          "items": {
            "type": "integer"
          }
        }
      },
      "required": ["task_id"]
    }
  },
  {
    "name": "task_list",
    "description": "List all tasks with status summary.",
    "input_schema": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "task_get",
    "description": "Get full details of a task by ID.",
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "integer"
        }
      },
      "required": ["task_id"]
    }
  }
]
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()


