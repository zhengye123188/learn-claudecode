import subprocess

from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID")
#Current Working Directory = cwd,告诉模型它所在的目录
SYSTEM = f"You are an coding agent at {os.getcwd()},Use bash to solve tasks,just do,not explain"
# 定义工具 Anthropic API 规定的格式
TOOLS = [{
    # 定义工具的名字
    "name" : "bash",
    # 工具的说明
    "description" : "Run a bash command",
    # 参数的结构定义
    "input_schema":{
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    },
}]
#输入字符串，返回字符串。为什么返回字符串？因为我们要把结果塞回 tool_result 喂给模型，模型吃的是文本。
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

# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,tools=TOOLS,max_tokens=8000)
        # 把模型的回复追加到历史里
        messages.append({"role":"assistant","content":response.content})
        # 检查是否继续
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            # block就是response.content当中的每个对象，例如TextBlock,ToolUseBlock
            # ToolUseBlock(
            #     type="tool_use",
            #     id="toolu_01ABC...",        # 唯一标识符
            #     name="bash",                 # 工具名
            #     input={"command": "ls -la"}  # 参数字典
            # )
            if block.type == "tool_use":
                # 取出该block的属性值的字段command，交给run_bash
                print(f"\033[33m$ {block.input['command']}\033[0m")  # 黄色显示命令
                output = run_bash(block.input["command"])
                print(output[:200]) #现在功能完整了，但你运行它只能看到最终答案，中间模型调了什么命令、返回了什么，全是黑箱。调试的时候会疯掉。
                # 是给 API 看的标签，告诉它"这个块要按工具结果处理，去匹配前面那个 tool_use"。
                # {"role": "user", "content": output}    # ❌ 模型会以为这是普通用户文本
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })
        messages.append({"role": "user", "content": results})


#只有当这个文件被直接运行时，才执行下面的代码。如果它被别的文件 import 进来，下面的代码不执行
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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

