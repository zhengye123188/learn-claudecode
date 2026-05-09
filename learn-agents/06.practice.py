import json
from dotenv import load_dotenv
load_dotenv()
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
MODEL=os.getenv("MODEL_ID")
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
WORKDIR = Path.cwd()
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# -- Tool implementations --
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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {
        "name": "compact",
        "description": "Trigger manual conversation compression.",
        "input_schema": {
            "type": "object",
            "properties": {
            "focus": {
                "type": "string",
                "description": "What to preserve in the summary"
                }
            }
        }
    }
]

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
}
def micro_compact(messages: list) -> list:
    tool_results = []
    for msg_idx,msg in enumerate(messages):
        if msg["role"] =="user" and isinstance(msg["content"], list):
            for part_idx,part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") =="tool_result":
                    tool_results.append((msg_idx,part_idx,part))
                    # tool_results内容大概如下
                    # tool_results = [
                    #     (2, 0,{"type": "tool_result", "tool_use_id": "toolu_A", "content": "main.py\nutils.py\nREADME.md"}),
                    #     (4, 0, {"type": "tool_result", "tool_use_id": "toolu_B", "content": "<main.py 的 1500 行>"}),
                    #     (6, 0,{"type": "tool_result", "tool_use_id": "toolu_C", "content": "1500 main.py\n200 utils.py"}),
                    # ]
                    #需要tool_use_id去上一个对话当中找出TOOLuseBlock中找出工具的name
    if len(tool_results) <= 3:
        return messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"]=="assistant" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.type == "tool_use":
                    tool_name_map[block.id] = block.name
    # 除了最后三个工具调用结果，其他工具调用结果的content全都覆盖成Previous tool result
    to_clear = tool_results[:-3]
    for _,_,result in to_clear:
        # 如果不是字符串或者长度比较小，可以不用压缩
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        result["content"] = f"[Previous tool_use{tool_name}]"
    return messages

def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]

def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4

def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > 50000:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = client.messages.create(model=MODEL,system=SYSTEM,tools=TOOLS,max_tokens=8000,messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        Compact=False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    Compact = True
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Compaction will run after this turn."
                    })
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
        # Layer 3: manual compact triggered by the compact tool
        if Compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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