import os
import subprocess
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
from agents.s02_tool_use import TOOL_HANDLERS
load_dotenv()
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL= os.getenv("MODEL_ID")
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."
TOOLs = [
    {
        "name":"bash",
        "description":"RUN a bash command",
        "input_schema":{
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
    },
    {
        "name":"read_file",
        "description":"Read file content",
        "input_schema":{
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["path"],
        },
    },
    {
        "name":"write_file",
        "description":"Write content to a file",
        "input_schema":{
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
        }
    },
    {
        "name":"edit_file",
        "description":"Edit file content",
        "input_schema":{
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_content": {"type": "string"},
                "new_content": {"type": "string"}
            },
            "required": ["path", "old_content", "new_content"],
        }
    }


]
# 防止模型跳出工作区外,确保path是工作区内
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command:str)->str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(command,shell=True, cwd=WORKDIR,
                               capture_output=True, text=True, timeout=120)
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        return output[0:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout 120s)"

def run_read(path:str, limit:int=None)->str:
    try:
        text = safe_path(path).read_text()
        lines = text.split()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...还剩{(len(lines)-limit)}行未读取"]
            return "\n".join(lines)[:50000] # 拼接行为字符串
    except Exception as e:
        return f"Error: {e}"

def run_write(path:str, content:str)->str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path, old_content, new_content):
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_content not in content:
            return f"Error: {old_content} not in {content}"
        else:
            fp.write_text(content.replace(old_content, new_content,1))
            return f"Edit {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash":lambda **kw:run_bash(kw["command"]),
    "read_file":lambda **kw:run_read(kw["path"], kw.get("limit")),
    "write_file":lambda **kw:run_write(kw["path"], kw["content"]),
    "edit_file":lambda **kw:run_edit(kw["path"], kw["old_content"],kw["new_content"]),
}

def agent_loop(messages: list):
    while True:
        response = client.messages.create(model=MODEL,messages=messages,system=SYSTEM,tools=TOOLs,max_tokens=8000)
        messages.append({"role":"assistant","content":response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type =="tool_use":
                # block就是response.content当中的每个对象，例如TextBlock,ToolUseBlock
                # ToolUseBlock(
                #     type="tool_use",
                #     id="toolu_01ABC...",        # 唯一标识符
                #     name="bash",                 # 工具名
                #     input={"command": "ls -la"}  # 参数字典
                # )
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}:")
                print(output[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role":"user","content":results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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


