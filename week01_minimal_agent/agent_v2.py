import json
import os
from openai import OpenAI

client = OpenAI(  # 初始化客户端和对话
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)


# 定义工具 json schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "calulate an arithmetic expression, like 2+3*4 or 2**3",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression, digits and + - * / ** ( ) only",
                    }
                },
                "required": ["expression"],  # 必填参数列表
            }
        }
    }
]

def calculator(expression: str) -> str:  # 定义函数
    import random
    # prob = random.random()  # just for Exercise3
    # if prob < 0.5:
    #     raise ValueError("other unkown error, just retry again")
    allowed = set("0123456789+-*/.() ")
    if not set(expression) <= allowed:
        # Returning an error message (not crashing) lets the model see what
        # went wrong and try again — exercise 3 explores this.
        raise ValueError(f"Expression contains disallowed characters: {expression!r}")
    return str(eval(expression))  # fine for a learning sandbox; never do this with untrusted input in production


TOOL_FUNCTIONS = {  # 工具名称--实际函数映射表
    "calculator": lambda inp: calculator(inp["expression"]),
}


MAX_ITERATIONS = 10    # 整个 agent loop 最多调用模型的轮数（防止无限循环）
MAX_TOOL_RETRIES = 3   # 同一个工具连续失败的最大次数


def run_agent(user_input: str) -> str:
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant. Use the tools when they apply."})  # 系统提示词
    messages.append({"role": "user", "content": user_input})  # 用户问题

    tool_failures = {}  # 每个工具"连续失败"的计数，成功一次就清零

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model="qwen3.7-plus",
            tools=TOOLS,
            messages=messages,
        )
        choice = response.choices[0]
        msg = choice.message

        # 关键一步：assistant 消息（包含 tool_calls）必须放回历史，
        # 否则下一轮请求模型就"忘了"自己刚才要调什么工具。
        messages.append(msg)

        if choice.finish_reason == "length":  # 输出被 max_tokens 截断
            return "[Response was truncated — raise max_tokens.]"

        if not msg.tool_calls:  # 没有工具调用 = 模型给出了最终答案
            return msg.content or "[no text in response]"

        print(f"[iteration {iteration}] {len(msg.tool_calls)} tool call(s)")
        for call in msg.tool_calls:
            name = call.function.name
            # 参数是 JSON 字符串；json.loads、查表、执行 都可能抛异常，
            # 统一 catch 后把错误字符串还给模型，而不是让程序崩溃。
            try:
                args = json.loads(call.function.arguments)
                print(f"  → tool call: {name}({call.function.arguments})")
                output = TOOL_FUNCTIONS[name](args)
                failed = False
                print(f"  ← result:    {output}")
            except Exception as e:
                output = f"Error: {e}"
                failed = True
                print(f"  ← error:     {e}")

            # 连续失败计数：达到上限后，在错误信息里明确告诉模型别再试了。
            # 就算模型不听话继续调，外层的 MAX_ITERATIONS 也兜底。
            if failed:
                tool_failures[name] = tool_failures.get(name, 0) + 1
                if tool_failures[name] >= MAX_TOOL_RETRIES:
                    output += (
                        f" (Tool '{name}' has now failed {tool_failures[name]} times in a row. "
                        "Do NOT call it again. Answer with what you already know, "
                        "or tell the user the tool is unavailable.)"
                    )
            else:
                tool_failures[name] = 0

            # 每个 tool_call 都必须有一条对应的 role="tool" 回复（id 要匹配），
            # 少答一个下一轮请求就会报 400。
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    # for 循环正常走完 = 模型在限定轮数内始终没给出最终答案
    return f"[Gave up: no final answer after {MAX_ITERATIONS} iterations.]"


if __name__ == "__main__":
    question = "What is 1234 * 5678, and what is 2 ** 16?"
    print(f"User: {question}\n")
    answer = run_agent(question)
    print(f"\nAgent: {answer}")


