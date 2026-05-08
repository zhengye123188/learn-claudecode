import os
import re
import subprocess
import yaml
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills" # 声明 skill 目录的位置

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

#---md文件
# name: git
# description: Git workflow helpers
# tags: vcs, workflow
# ---
#
# # Git Workflow
#
# Step 1: Always create a feature branch from main.
# Step 2: Run pre-commit hooks before pushing.
# Step 3: Use conventional commits format.

# meta:
# {
#     "name": "git",
#     "description": "Git workflow helpers",
#     "tags": "vcs, workflow"
# }
#body:
#"# Git Workflow\n\nStep 1: Always create a feature branch from main.\nStep 2: Run pre-commit hooks before pushing.\nStep 3: Use conventional commits format."
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def load_skills(self):
        if not self.skills_dir.exists():
            return
        for file in sorted(self.skills_dir.rglob("SKILL.md")):
            text = file.read_text().strip()
            meta,body = self.parse_md_frontmatter(text)
            name = meta.get("name",file.parent.name)
            self.skills[name] = {"meta": meta, "body": body,"path": str(file)}

    def parse_md_frontmatter(self, text)->tuple[str, str]:
        """
            传入 Markdown 文本内容，解析 frontmatter meta 和正文 body
            :param text: md 完整文本
            :return: (meta字典, body正文)
            """
        # 匹配开头 --- 包裹的 frontmatter
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)"
        match = re.match(pattern, text, re.DOTALL | re.MULTILINE)
        if not match:
            return {}, text.strip()
        yaml_str = match.group(1)
        body = match.group(2).strip()
        try:
            meta = yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, body

    def get_simple_description(self) -> str:
        """
        Layer 1: Build a short skill index for the system prompt.

        Returns only skill names and descriptions (from frontmatter meta),
        NOT full body content. The full body is loaded on demand via
        get_skill_body() when the model calls load_skill.

        :return: formatted system prompt string
        """
        if not self.skills:
            return "No available skills currently."

        skill_lines = []
        for name, skill_info in self.skills.items():
            meta = skill_info.get("meta", {})
            desc = meta.get("description", "No description")
            tags = meta.get("tags", "")

            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            skill_lines.append(line)

        content = "\n".join(skill_lines)
        return f"Available Skills:\n{content}"

    def get_skill_body(self,name:str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


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
        "name": "load_skill",
        "description": "Load a skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of skill.",
                },
            },
            "required": ["name"]
        },
    }
]

SKILL_LOADER = SkillLoader(SKILLS_DIR)
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_skill_body(kw["name"]),
}

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.
Skills available:
{SKILL_LOADER.get_simple_description()}"""
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
            query = input("\033[36ms05 >> \033[0m")
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