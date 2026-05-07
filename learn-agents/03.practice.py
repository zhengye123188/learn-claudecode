import os
import subprocess
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL= os.getenv("MODEL_ID")
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""



def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command:str)->str:
    # 防止危险命令被执行
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # command要执行的命令
        # shell=True让字符串通过 shell解释
        # capture_output=True把子进程的 stdout 和 stderr 抓下来，而不是打到终端
        # text=True返回字符串而不是字节流（不然要自己 .decode('utf-8')
        # subprocess.run 返回一个 CompletedProcess 对象，里面有 stdout、stderr、returncode 等字段。

        # timeout防止挂死，模型跑了个 sleep 999999，或者不小心启动了一个交互式程序（python 开 REPL、vim 打开编辑器），或者 ping google.com（不加 -c 会无限 ping）。
        # 这些命令永远不会结束，你的 agent 就卡死了
        result = subprocess.run(command,shell=True, cwd=os.getcwd(),
                               capture_output=True, text=True, timeout=120)
        out = (result.stdout+result.stderr).strip()
        # 有些命令成功了但没有任何输出，比如 cd /tmp、touch foo.txt。这时 out 是空字符串 ""。
        # 假设command是cat，获取文件所有文字，文字内容过大需要截取，否则会超出上下文限制
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Timed out(120s)"
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e}"
    except FileNotFoundError as e:
        return f"FileNotFoundError: {e}"

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

class TodoManager:
    def __init__(self):
        self.items = []

    def render(self)->str:
        if not self.items:
            return "no todos"
        lines = []
        mapping = {
            "pending":"[ ]",
            "in_progress":"[>]",
            "completed":"[X]",
        }
        for item in self.items:
            maker = mapping[item["status"]]
            lines.append(f"任务id:{item["id"]},任务内容:{item["text"]}，任务状态:{maker}。")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def update(self, items:list)->str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []  # 暂存验证通过的项,这是事务性思维。想象循环到第 5 项时发现状态拼错了,抛异常。如果我们一边循环一边写 self.items,那 self.items 现在就只有前 4 项,数据被部分破坏了。
        # self.items = [
        #     {"id": "1", "text": "读 hello.py", "status": "completed"},
        #     {"id": "2", "text": "加类型注解", "status": "in_progress"},
        #     {"id": "3", "text": "加 docstring", "status": "pending"},
        # ]
        in_progress_count = 0 #同时只能一个 in_progress(全文最重要的一条)
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id} has no text")
            if status not in ["pending", "in_progress", "completed"]:
                raise ValueError(f"Item {item_id} has an invalid status: {status}")
            if status == "in_progress":
                in_progress_count += 1
            # 通过了所有单项检查,加入临时列表
            validated.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

TODO = TodoManager()
TOOLS = [
    {
        "name":"bash",
        "description": "run a bash command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                }
            },
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
    },
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["id", "text", "status"]
                    }
                }
            },
            "required": ["items"]
        }
    }
]
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

def agent_loop(messages:list)->str:
    # 如果三个回合没有更新任务清单，提醒agent调用todo.update工具
    round_since_todo = 0
    while True:
        response = client.messages.create(model=MODEL, messages=messages, system=SYSTEM, tools=TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
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
                if block.name == "todo":
                    used_todo = True
        round_since_todo = 0 if used_todo else round_since_todo + 1
        if round_since_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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


