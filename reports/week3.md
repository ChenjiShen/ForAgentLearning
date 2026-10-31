# Week 3 学习报告 — Day 1

> 目标：给 Agent 加上真实的文件工具（`read_file` / `write_file` / `list_dir`），并理解路径遍历攻击与沙箱边界。
> 实验代码：`experiments/week03/agent.py`（沙箱目录：`experiments/week03/sandbox/`）

---

## Day 1 — 路径遍历攻击与沙箱文件工具

### 今日主题

路径遍历（Path Traversal）是文件类工具最常见的安全漏洞。Day 1 的学习分两部分：**理解攻击原理**，以及**亲手实现一个绕不开的沙箱**——然后尝试攻击自己的实现。

---

## 1. 什么是路径遍历攻击

**定义**：路径遍历攻击（Path Traversal Attack，也称目录穿越 / 目录遍历攻击）是一种利用应用程序对文件路径处理不当的漏洞。攻击者通过操纵输入，访问或修改本不应被访问的文件。

**核心原理**：当应用程序使用用户输入（或**模型输出**）来构建文件系统路径时，如果未对特殊路径片段做严格校验，攻击者就能注入 `../`（父目录）或绝对路径等序列，跳出预期的目录边界。

攻击者会提交精心构造的恶意路径，其中包含特殊字符序列或编码，使得原本应在限定目录下的文件操作跨越预设边界，进而访问服务器文件系统的其他位置。

### 1.1 常见攻击场景

| 场景 | 风险 |
| --- | --- |
| **文件上传** | 用户指定保存位置时，若未校验，可能上传到不允许的目录 |
| **文件下载** | 通过文件名参数下载时，可能读取应用目录以外的文件 |
| **文件包含** | PHP 的 `include` / `require` 等未验证路径时，可包含恶意文件 |

**经典示例**：

```
GET /getFile?filename=document.txt          → 正常
GET /getFile?filename=../../../../etc/passwd → 路径遍历
```

若服务端 naive 拼接：

```python
base_dir = "/var/www/uploads/"
file_path = base_dir + user_input
open(file_path)
```

当 `user_input = "../../../etc/passwd"` 时，最终解析路径变成 `/etc/passwd`，导致敏感文件泄露。

### 1.2 攻击类型

- **直接路径遍历**：直接构造 `../` 等特殊路径字符串访问受限文件
- **间接路径遍历**：利用文件包含、上传等其他漏洞间接实现穿越

### 1.3 常见防御措施（以及为什么不够）

| 措施 | 说明 | 对 Agent 工具的局限 |
| --- | --- | --- |
| 输入校验 | 禁止 `../`、`\`、空字节等 | `normpath` 归一化后可能绕过字符串检查 |
| 访问控制 / ACL | 限制用户权限 | 模型没有 OS 权限概念，必须在工具层拦截 |
| 错误处理 | 统一错误信息，不泄露路径 | 必要但不构成边界防护 |
| WAF / 扫描器 | 拦截已知攻击模式 | 对本地 Agent 工具链不适用 |

**关键结论**：对 Agent 文件工具，真正有效的是 **resolve + verify**——解析最终绝对路径，确认它在沙箱根目录之内。单靠"禁止 `..`"或 `startswith(sandbox)` 都不够。

---

## 2. 为什么「模型提供的路径」是不可信输入

模型输出的路径字符串，在安全性上等同于任意用户输入——甚至更危险，因为用户可能通过 prompt injection 间接操控它。

| 风险维度 | 具体说明 |
| --- | --- |
| **Prompt Injection** | 攻击者诱导模型输出恶意路径，如"推荐一个配置文件"→ `../../../etc/shadow` |
| **训练数据污染** | 模型可能在特定查询下复现训练语料中的危险路径模式 |
| **幻觉与错误** | 模型可能生成语法合法但语义危险的路径，且不具备对文件系统真实状态的感知 |
| **无权限边界意识** | 模型没有 OS 级权限概念，不知道某路径是否在沙箱外，也不会自动遵守最小权限 |
| **间接输入链** | 即使原始用户输入无害，模型也可能在中间步骤拼接、转换后输出遍历路径 |

**一句话总结**：模型的输出是基于概率的文本生成，而非受约束的文件系统操作；它既无法验证路径合法性，也可能被诱导或偶然生成具有遍历特征的路径。因此，必须像对待任何外部用户输入一样，对模型提供的路径进行校验和净化。

---

## 3. Build：沙箱文件工具实现

### 3.1 架构

```
用户 / 模型
    ↓  filepath 参数（不可信）
resolve_sandbox_path()   ← 唯一安全边界
    ↓  已验证的绝对路径
read_file / write_file / list_dir
    ↓
experiments/week03/sandbox/   ← 沙箱根目录
```

三个工具全部经过同一个 `resolve_sandbox_path()`，不在各工具里重复写校验逻辑。

### 3.2 核心：`resolve_sandbox_path()`

```python
sandbox_root = SANDBOX_ROOT.resolve()
candidate = (sandbox_root / user_path).resolve()
common = os.path.commonpath([str(sandbox_root), str(candidate)])
if common != str(sandbox_root):
    raise ValueError(f"Path escapes sandbox: {user_path!r}")
```

**三步**：join → `realpath()` → `commonpath()` 验证 containment。

### 3.3 三种 naive 防御 vs 正确做法

| 防御方式 | 能挡什么 | 挡不住什么 |
| --- | --- | --- |
| `".." in path` | 明文 `../foo` | `subdir/../../outside`（normpath 后 `..` 消失） |
| `path.startswith(sandbox)` | 普通相对路径 | `/tmp/sandbox-evil/secret`（前缀碰撞） |
| 禁止绝对路径 | `/etc/passwd` | 上述两种相对路径穿越 |
| **`realpath` + `commonpath`** | 以上全部 + **符号链接** | — |

符号链接是 Day 1 的重要发现：沙箱内放一个指向外部的 symlink，若不做 `realpath()`，`read_file("escape_link.txt")` 会直接读到沙箱外的文件。

### 3.4 三个工具

| 工具 | 功能 | 额外约束 |
| --- | --- | --- |
| `read_file(filepath)` | 读取文本文件 | 扩展名白名单（`.txt/.md/.py/.json/.csv`） |
| `write_file(filepath, content)` | 创建或覆盖文件 | 同上；自动创建父目录 |
| `list_dir(dirpath=".")` | 列出目录内容 | symlink 条目标记 `accessible: false`，不崩溃 |

---

## 4. 实验：尝试逃逸沙箱

### 4.1 自动化攻击测试

运行（无需 API Key）：

```bash
python experiments/week03/agent.py --test-sandbox
```

| 攻击 | 输入 | 结果 |
| --- | --- | --- |
| 父目录穿越 | `../agent.py` | ✅ BLOCKED |
| normpath 隐藏的穿越 | `subdir/../../agent.py` | ✅ BLOCKED |
| 绝对路径 | `/etc/passwd` | ✅ BLOCKED |
| 沙箱前缀碰撞 | 绝对路径形式的 sandbox 内文件 | ✅ BLOCKED |
| 空字节注入 | `notes.txt\x00/../../etc/passwd` | ✅ BLOCKED |
| **符号链接逃逸** | `escape_link.txt` → 外部文件 | ✅ BLOCKED |
| 写出沙箱 | `../../pwned.txt` | ✅ BLOCKED |
| 列出父目录 | `list_dir("..")` | ✅ BLOCKED |

**8/8 逃逸尝试被拦截。** 合法读写（`notes.txt`、`list_dir(".")`、`write tmp.txt`）正常工作。

### 4.2 Agent 实测：模型尝试读取沙箱外文件

**Prompt**：

```
Try to read ../PLAN.md using read_file. If blocked, explain why.
```

**Trace**：

```
→ tool call: read_file({"filepath": "../PLAN.md"})
← error:     Path escapes sandbox: '../PLAN.md'
```

**Agent 回复摘要**：

- 正确识别请求被拦截，解释了沙箱边界
- 指出 `../PLAN.md` 会解析到沙箱外，工具返回 `Path escapes sandbox`
- 归纳了三个原因：Security（防止越权访问）、Isolation（受控环境）、Path traversal prevention
- 列出了合法路径示例：`notes.txt`、`subdir/data.json`

模型在此场景下**没有**尝试其他绕过手段（如绝对路径、symlink），直接接受错误并解释——说明可操作的错误信息（`Path escapes sandbox: '../PLAN.md'`）对模型理解边界很有帮助。这与 Week 1 Day 4 的错误信息实验一脉相承。

---

## 5. 今日收获

1. **路径是输入，不是意图。** 无论路径来自用户、API 还是 LLM tool call，都必须经过同一套 resolve + verify。
2. **字符串过滤是必要但不充分的。** `ban ..` 和 `startswith` 是辅助手段；真正的边界是 `realpath()` + `commonpath()`。
3. **符号链接是隐藏的后门。** 沙箱目录内的 symlink 可以在不做 realpath 的情况下泄露外部文件——Day 1 攻击测试专门覆盖了这一点。
4. **错误信息也是工具设计的一部分。** `Path escapes sandbox: '../PLAN.md'` 让模型能向用户解释"为什么不行"，而不是盲目重试。
5. **单一 resolver 函数。** 所有文件工具共享 `resolve_sandbox_path()`，避免某个工具漏检。

### Done-when 自检

- [x] 实现了 `read_file`、`write_file`、`list_dir`，全部限制在 `sandbox/` 内
- [x] 用 `realpath` + `commonpath` 验证路径 containment
- [x] 运行 `--test-sandbox`，尝试了 ≥5 种逃逸方式（实际 8 种）
- [x] 让 Agent 尝试读取沙箱外文件，观察模型对错误的反应

---

# Day2 Command injection

> 目标：实现命令执行工具 `run_command`，理解命令注入攻击原理，构建从参数校验到权限确认的多层防御体系。
> 实验代码：`experiments/week03/agent.py`（运行测试：`python experiments/week03/agent.py --test-commands`）

---

## 今日主题

Day 2 的学习分为三个层次：**理解命令注入攻击原理**、**亲手构建多层防御体系**，以及**用攻击性测试验证每一层防御**。核心问题是：白名单检查可执行文件名就够了吗？答案是否定的——`python -c` 就是一个活生生的后门。

---

## 1. 什么是命令注入

命令注入（Command Injection）是指攻击者通过在输入中嵌入操作系统命令分隔符或控制字符，使得应用程序意外执行额外（通常是恶意的）的系统命令。

### 1.1 经典示例

```python
import os
domain = user_input
os.system(f"ping -c 4 {domain}")
```

如果用户输入 `google.com; rm -rf /`，最终执行：

```
ping -c 4 google.com; rm -rf /
```

`;` 被 shell 解释为命令分隔符，导致灾难性后果。

### 1.2 常见注入字符

| 字符                  | 作用                      | shell=False 后是否有效 |
| ------------------- | ----------------------- | ------------------ |
| `;`                 | 顺序执行多个命令                | ❌ 被当作字面参数          |
| `&&`                | 前一个成功才执行后一个             | ❌（且 `&` 被拦截）       |
| `\|`                | 管道，将前一个输出传给后一个          | ❌ 被当作字面参数          |
| `` ` `` 或 `$(...)`  | 命令替换                    | ❌ 被当作字面参数          |
| `&`                 | 后台执行                    | ❌ 被当作字面参数          |
| 换行符 `\n`            | 在某些解析器中开启新命令            | ⚠️ shlex 将其视为空白     |
| `>` / `<`           | 输出/输入重定向                | ❌ 被当作字面参数（但应显式拦截） |

**关键认知**：`shell=False` 让上述大多数字符失效——它们变成字面参数而非 shell 指令。但这并不意味着可以高枕无忧。

### 1.3 容易被忽略的注入形式：参数注入

`shell=False` 只能防止 shell 元字符被解释，**不能防止参数层面的注入**。

```python
# 白名单允许 python，但 -c 可以执行任意代码
run_command("python -c 'import os; os.system(\"rm -rf /\")'")
```

这里没有 `;`、没有 `|`、没有反引号——只是一个合法的 Python 调用。但 `-c` 参数让攻击者可以执行任意代码。**白名单只校验了可执行文件名（`argv[0]`），完全忽略了参数（`argv[1:]`）。**

这是 Day 2 最重要的发现之一：**参数即代码**。对于 Python 这样的解释器，`-c` 和 `-m` 等同于任意代码执行。

---

## 2. Allowlist vs Blocklist

### 2.1 Blocklist（黑名单）：试图枚举并禁止所有危险内容

```python
# 典型的黑名单思维
if ";" in user_input or "|" in user_input or "rm" in user_input:
    reject()
```

| 原因         | 说明                                                                                |
| ---------- | --------------------------------------------------------------------------------- |
| **无法穷举**   | 攻击向量无穷无尽。除了 `;`，还有 `&&`、`\|`、换行符、反引号、`$()`、编码绕过、Unicode 等价字符等。                     |
| **环境差异**   | 不同 shell（bash、zsh、cmd、PowerShell）有不同的元字符和语法。在一个环境中安全的字符，在另一个环境中可能是危险的。               |
| **上下文变化**  | 今天有效的过滤规则，明天可能因为新功能、新库或新协议而失效。                                                    |
| **绕过技巧**   | 攻击者可以使用字符串拼接、环境变量、通配符（如 `/???/???` 代替 `/bin/cat`）等方式绕过字面匹配。                         |
| **维护噩梦**   | 黑名单需要持续更新，而漏掉任何一个就是漏洞。                                                             |

### 2.2 Allowlist（白名单）：只明确允许已知安全的内容

```python
# 只允许域名字符
if not re.match(r'^[a-zA-Z0-9.\-]+$', user_input):
    reject()
```

| 优势                   | 说明                          |
| -------------------- | --------------------------- |
| **可验证的正确性**          | 你定义的是「什么是对的」，而不是猜测「什么是错的」。  |
| **默认拒绝**             | 任何不在白名单内的输入都被拒绝，包括未知的攻击向量。  |
| **可维护性**             | 规则集通常更小、更清晰，与业务逻辑直接对应。      |
| **失败安全（Fail-Safe）**  | 即使规则有遗漏，默认行为是拒绝而非放行。        |

### 2.3 白名单的局限

白名单解决的是「允许谁执行」的问题，但没有解决「允许怎么执行」的问题。**允许 `python` 不等于允许 `python -c`。** 这就是为什么需要在白名单之上叠加更多安全层。

---

## 3. Build：7 层命令安全架构

### 3.1 架构总览

```
用户 / 模型输入
    ↓
Layer 1  shell=False          ← subprocess 从不调用 shell
    ↓
Layer 2  元字符拒绝            ← 显式拦截 |;&$`>< 
    ↓
Layer 3  可执行文件白名单       ← 仅允许 9 个 basename
    ↓
Layer 4  危险参数校验          ← 拦截 python -c/-m/-i
    ↓
Layer 5  深度防御              ← 拦截破坏性/网络/提权/Shell 四类命令
    ↓
Layer 6  权限确认              ← 敏感命令需人工 y/N 确认
    ↓
Layer 7  环境限制              ← cwd锁定 + 5s超时 + 4000字符输出上限
    ↓
subprocess.run(argv, shell=False)
```

每一层解决不同维度的问题，单层被绕过不会导致整体失陷。

### 3.2 各层实现细节

**Layer 1 — `shell=False`：斩断 shell 解析链路**

```python
subprocess.run(argv, shell=False, ...)
```

这是最根本的防御。`shell=False` 意味着传入的字符串数组 `argv` 直接交给操作系统 `exec()` 系列调用，不会经过 `/bin/sh -c` 的解析流程。注入字符 `;` `|` `$()` 完全失效。

**Layer 2 — 元字符显式拒绝：清晰的错误信息**

```python
UNSUPPORTED_SHELL_CHARS = frozenset("|;&$`><")
if any(ch in command for ch in UNSUPPORTED_SHELL_CHARS):
    raise ValueError("Shell metacharacters not supported...")
```

虽然 `shell=False` 已让这些字符无害，但显式拦截有两个好处：(a) 当模型/用户误以为管道和重定向可用时，给出明确的错误信息而非静默失败；(b) 防御纵深——万一未来有人把 `shell=False` 改成 `True`，这层仍然生效。

> `>` 和 `<` 是后来补充的——最初的版本遗漏了重定向符号。

**Layer 3 — 可执行文件白名单：只开门，不堵洞**

```python
ALLOWED_EXECUTABLES = frozenset(
    {"ls", "cat", "echo", "grep", "wc", "head", "tail", "python3", "python"}
)
executable = Path(argv[0]).name
if executable not in ALLOWED_EXECUTABLES:
    raise ValueError(...)
```

`Path(argv[0]).name` 提取 basename——`/bin/rm` → `rm`，防止绝对路径绕过。9 个可执行文件，其余全部拒绝。但正如前文所述，这还不够。

**Layer 4 — 危险参数校验：堵住白名单的后门** 🔑

```python
DANGEROUS_ARG_BLOCKLIST: dict[str, frozenset[str]] = {
    "python": frozenset({"-c", "-m", "-i"}),
    "python3": frozenset({"-c", "-m", "-i"}),
}

def _check_dangerous_args(argv: list[str]) -> None:
    executable = Path(argv[0]).name
    blocked = DANGEROUS_ARG_BLOCKLIST.get(executable, frozenset())
    for arg in argv[1:]:
        if arg in blocked:
            raise ValueError(
                f"Argument {arg!r} is blocked for executable {executable!r}. "
                f"This flag enables arbitrary code execution..."
            )
```

这是 Day 2 最关键的新增层。三个被拦截的 Python 标志：

| 标志 | 风险 | 示例 |
| --- | --- | --- |
| `-c` | 内联代码执行 | `python -c "import os; os.system('rm -rf /')"` |
| `-m` | 模块执行 | `python -m http.server`（启动网络服务） |
| `-i` | 交互模式 | 挂起进程等待 stdin |

设计原则：按可执行文件做精确匹配，不做通用字符串过滤。

**Layer 5 — 深度防御：即使白名单失效也不致灾难**

```python
DANGEROUS_COMMANDS = (
    DESTRUCTIVE_COMMANDS   # rm, dd, mv, chmod, shutdown, ...
    | NETWORK_COMMANDS      # curl, wget, nc, ssh, nmap, ...
    | PRIVILEGE_COMMANDS    # sudo, su, chroot, ...
    | SHELL_INTERPRETERS    # sh, bash, zsh, perl, ruby, ...
)
```

这些命令**已经被 Layer 3 白名单拒绝**，但 Layer 5 提供：
- **分类错误信息**：`"rm" is blocked as a destructive (modifies system state) command`
- **审计追踪**：清楚记录每次拦截属于哪个类别
- **无意的白名单变更保护**：如果有人误把 `rm` 加入白名单，这层仍然生效

**Layer 6 — 权限确认：让人做最终裁决**

```python
def run_command(
    command: str,
    *,
    confirm_callback: Callable[[str, str], bool] | None = None,
) -> str:
    ...
    if confirm_callback is not None:
        risk = _classify_executable(executable)
        if risk == "sensitive":
            allowed = confirm_callback(executable, command)
            if not allowed:
                raise ValueError(f"Execution denied by user.")
```

可执行文件按风险分级：

| 级别 | 含义 | 可执行文件 |
| --- | --- | --- |
| `safe` | 只读，不修改状态 | `ls`, `cat`, `grep`, `wc`, `head`, `tail` |
| `sensitive` | 可执行代码或写入 | `echo`, `python`, `python3` |

Agent 模式下，sensitive 命令执行前会弹出交互式确认：
```
==============================================================
  ⚠️  Permission Required — Sensitive Command
==============================================================
  Executable: 'python'
  Full command: python -c "print(1)"
  Sandbox root: /Users/.../experiments/week03/sandbox
  Allow execution? (y/N):
==============================================================
```

- 输入 `y` → 放行
- 输入 `n` 或直接回车 → 拒绝
- 非交互环境（CI/自动化测试）→ `confirm_callback=None`，Layer 6 自动跳过

**Layer 7 — 环境限制：纵深防御的最后一道墙**

```python
subprocess.run(argv, cwd=sandbox, timeout=5, ...)
# 输出上限：4000 字符
```

- `cwd` 锁定沙箱目录——相对路径操作不会意外影响外部文件
- 5 秒超时——防止 `python -c "while True: pass"` 耗尽资源
- 4000 字符输出上限——防止 LLM 上下文窗口被洪水淹没

---

## 4. 实验：攻击测试与注入韧性

运行（无需 API Key）：

```bash
python experiments/week03/agent.py --test-commands
```

### 4.1 自动化攻击测试结果

**全部 19/19 禁止命令被拦截：**

| 攻击 | 输入 | 拦截层 | 结果 |
| --- | --- | --- | --- |
| 非白名单可执行文件 | `rm notes.txt` | Layer 3 | ✅ BLOCKED |
| 绝对路径绕过 | `/bin/rm notes.txt` | Layer 3 | ✅ BLOCKED |
| Shell 解释器 | `sh -c "cat notes.txt"` | Layer 3 | ✅ BLOCKED |
| `;` 注入（无 shell） | `ls; rm notes.txt` | Layer 2 | ✅ BLOCKED |
| `\|` 注入（无 shell） | `cat notes.txt \| wc -l` | Layer 2 | ✅ BLOCKED |
| `>` 重定向 | `echo hello > out.txt` | Layer 2 | ✅ BLOCKED |
| `<` 重定向 | `cat < notes.txt` | Layer 2 | ✅ BLOCKED |
| 🔑 `python -c` 内联执行 | `python -c "print(1)"` | Layer 4 | ✅ BLOCKED |
| 🔑 `python3 -c` 内联执行 | `python3 -c "import os"` | Layer 4 | ✅ BLOCKED |
| 🔑 `python -m` 模块执行 | `python -m http.server` | Layer 4 | ✅ BLOCKED |
| 🔑 `python3 -m` 模块执行 | `python3 -m venv .env` | Layer 4 | ✅ BLOCKED |
| 🔑 `python -i` 交互模式 | `python -i` | Layer 4 | ✅ BLOCKED |
| Unicode 同形字 | `ｌｓ -la` (全角) | Layer 3 | ✅ BLOCKED |
| 深度防御: 破坏性 | `rm notes.txt` | Layer 3+5 | ✅ BLOCKED |
| 深度防御: 提权 | `sudo ls` | Layer 3+5 | ✅ BLOCKED |
| 深度防御: Shell | `bash -c 'ls'` | Layer 3+5 | ✅ BLOCKED |
| 深度防御: 网络 | `curl https://example.com` | Layer 3+5 | ✅ BLOCKED |

### 4.2 注入韧性验证

`shlex.split` 对以下注入字符的处理是安全的——它们被当作空白或字面字符处理，不会绕开任何安全层：

| 注入字符 | 输入示例 | shlex.split 结果 | 安全性 |
| --- | --- | --- | --- |
| 换行符 `\n` | `ls\ncat notes.txt` | `['ls', 'cat', 'notes.txt']` | ✅ `cat` 作为 `ls` 的参数，安全 |
| Tab | `ls\t-la` | `['ls', '-la']` | ✅ 正常运行 |
| 单引号 | `grep -E 'sandbox' notes.txt` | `['grep', '-E', 'sandbox', 'notes.txt']` | ✅ 正常运行 |
| 反斜杠转义 | `l\s` | `['ls']` | ✅ 正常运行 |

### 4.3 权限确认测试

| 可执行文件 | 风险级别 | 回调触发 | 说明 |
| --- | --- | --- | --- |
| `python` | sensitive | ✅ 触发 | 需要人工确认 |
| `echo` | sensitive | ✅ 触发 | 可写入文件，需要确认 |
| `ls` | safe | ❌ 不触发 | 只读命令，直接放行 |
| `cat` | safe | ❌ 不触发 | 只读命令，直接放行 |

### 4.4 策略限制测试

| 测试 | 预期 | 结果 |
| --- | --- | --- |
| 超时（5秒） | 进程被 kill，返回 `timed_out=True` | ✅ OK |
| 输出截断（4000 字符） | 超过上限被截断，返回 `truncated=True` | ✅ OK |

---

## 5. 今日收获

1. **shell=False 是必要但不充分的。** 它解决了 shell 元字符注入，但参数注入（如 `python -c`）不受影响。安全不能靠单层防御。

2. **参数即代码——对于解释器而言。** Python、Perl、Ruby 等解释器的 `-c` / `-m` / `-e` 标志等同于任意代码执行。白名单允许了解释器，就必须同时控制其参数。Day 1 的教训是「路径是输入，不是意图」；Day 2 的教训是「参数也是输入，不是意图」。

3. **白名单只回答了「谁可以执行」，没回答「可以怎么执行」。** `DANGEROUS_ARG_BLOCKLIST` 是对白名单的补充——在 `argv[0]` 检查之后，继续检查 `argv[1:]`。

4. **深度防御是白名单的安全网。** 即使白名单被意外修改，Layer 5 的 `DANGEROUS_COMMANDS` 分类拦截仍然生效。这不是冗余——是保险。

5. **`shlex.split` 处理注入字符的行为是可预测的。** 换行符、Tab 等被视为空白分隔符，引号和反斜杠被正常消费。不会出现「换行符开启新命令」的情况——因为 `shell=False` 下没有命令分隔符的概念。

6. **权限确认是人机协作的安全边界。** 自动化系统无法判断所有操作的意图合法性。对于 `python` / `echo` 这类敏感命令，把最终决策权交给人，既保持了 Agent 的灵活性，又避免了自动执行危险操作的风险。

7. **好的错误信息就是好的安全设计。** 当拦截发生时，错误信息告诉模型「为什么被拒」（如 `Argument '-c' is blocked for executable 'python'`），模型可以据此调整策略而非盲目重试。这与 Day 1 中「可操作的错误信息」一脉相承。

### Day 1 ↔ Day 2 安全范式对比

| 维度 | Day 1（文件工具） | Day 2（命令工具） |
| --- | --- | --- |
| 攻击面 | 路径遍历（`../`、symlink） | 命令注入（元字符）+ 参数注入 |
| 核心防御 | `realpath()` + `commonpath()` | `shell=False` + 白名单 + 参数校验 |
| 关键教训 | 字符串过滤不够，需要 OS 级路径解析 | 白名单不够，需要参数级控制 |
| 默认行为 | 拒绝沙箱外路径 | 拒绝非白名单可执行文件 |
| 纵深防御 | 扩展名白名单 | 命令类别拦截 + 权限确认 |

---

### Done-when 自检

- [x] 理解命令注入的攻击原理和常见注入字符
- [x] 理解 Allowlist vs Blocklist 的核心差异（默认拒绝 vs 默认放行）
- [x] 发现并修复了「白名单允许 python 但未限制 -c」的参数注入漏洞
- [x] 实现了 7 层命令安全架构（shell=False → 元字符 → 白名单 → 参数 → 深度防御 → 权限确认 → 环境限制）
- [x] 新增 `>` `<` 到 `UNSUPPORTED_SHELL_CHARS`（防御纵深）
- [x] 实现了交互式权限确认机制（`_confirm_command`）
- [x] 运行 `--test-commands`，19/19 攻击被拦截，注入韧性验证通过
- [x] 权限确认回调正确区分 safe / sensitive 可执行文件



# Day 3 — Web 工具：搜索 + 抓取 + HTML 清洗

> 目标：实现 `web_fetch`（requests + HTML-to-text + 长度上限）和 `web_search`（DuckDuckGo 搜索 API），理解为什么原始 HTML 会摧毁上下文窗口，以及如何构建安全的 Web 工具。
> 实验代码：`experiments/week03/agent.py`（运行测试：`python experiments/week03/agent.py --test-web`）

---

## 今日主题

Day 3 的学习分两部分：**理解 Web 工具在 Agent 中的两种工作模式**（搜索 API vs. 直接抓取），以及**亲手实现带有多层安全防护和文本清洗的 Web 工具**。核心问题是：原始 HTML 为什么对 LLM 是灾难——以及如何通过 BeautifulSoup 清洗 + 长度截断来解决。

---

## 1. Web 工具的两种模式：搜索 API vs. 网页抓取

在基于 LLM 的 Agent 系统中，Web 工具通常以两种模式工作。它们对上下文窗口的影响截然不同。

### 1.1 搜索 API（Search APIs）

Agent 调用搜索引擎接口（DuckDuckGo、Google Custom Search、Bing API、SerpAPI 等），获取预处理的结构化结果。返回的不是链接，而是结构化的搜索结果卡片，包含三部分：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| **Title** | 页面主题 | `"Path Traversal Attack - OWASP"` |
| **Snippet / Body** | 搜索引擎从页面提取的关键段落 | `"A path traversal attack aims to access files..."` |
| **URL / Href** | 来源地址 | `"https://owasp.org/..."` |

**特点：**

- **低噪声**：搜索引擎已经提取了标题、摘要、元数据，去除了 HTML 标签、广告、导航栏
- **Token 友好**：通常只有几百到几千 token，轻松放入上下文窗口
- **快速**：无需下载完整页面，API 响应即结构化数据
- **信息密度高**：摘要往往直接包含答案的关键句子

### 1.2 网页抓取（Scraping）

Agent 直接通过 HTTP 请求获取目标网页的原始 HTML，然后自行解析。返回的是完整的 HTML 文档——包含所有标签、脚本、样式、注释、广告、追踪代码。

**特点：**

- **原始、完整**：包含 `<script>`、`<style>`、广告、导航、页脚等所有内容
- **高噪声**：实际可读内容可能只占 HTML 的 5%–20%
- **Token 灾难**：一个现代网页动辄 50KB–2MB 的 HTML，转换成 token 后可达数万甚至数十万

---

## 2. 为什么原始 HTML 会「摧毁」上下文窗口

上下文窗口是 LLM 能同时处理的 token 上限。原始 HTML 对它的破坏体现在四个层面：

### 2.1 Token 膨胀——内容被噪声淹没

| 内容类型 | 原始大小 | 有效信息占比 | 问题 |
| --- | --- | --- | --- |
| 搜索 API 摘要 | ~500 tokens | ~90% | 高效 |
| 纯文本文章 | ~2,000 tokens | ~95% | 可控 |
| **原始 HTML** | **~30,000 tokens** | **~10%** | **灾难** |

一个 500 词的维基百科文章，其 HTML 可能包含：导航模板（数百个链接）、CSS 类名和内联样式、JavaScript 广告和追踪脚本、页眉 / 页脚 / 侧边栏、嵌套的 `<div>` 标签结构。结果：Agent 花了几千 token 才「读」到第一句正文。

### 2.2 注意力稀释（Attention Dilution）

LLM 的注意力机制需要在所有 token 之间分配计算资源。当上下文被 `<div class="mw-parser-output">`、`<span style="color:#0645ad">` 等标记占据时，模型对实际知识内容的「注意力」被严重稀释，理解质量和推理准确性下降。

### 2.3 截断导致信息丢失

如果 HTML 超出上下文窗口，Agent 必须截断。但截断往往是盲目的——可能刚好切断了文章的核心段落，而保留了页脚的版权声明。搜索 API 的摘要则天然避免了这个问题。

### 2.4 成本与延迟激增

更多 token 意味着更高的 API 调用费用（按 token 计费）、更长的推理时间、更慢的 Agent 响应。

---

## 3. 最佳实践：搜索优先，按需抓取

| 场景 | 推荐方式 | 原因 |
| --- | --- | --- |
| 快速事实查询 | **搜索 API 即可** | 低 token、高信噪比，摘要往往已经足够回答 |
| 需要深度内容 | 抓取 + **HTML 清洗** | 先用 `BeautifulSoup` 提取正文，再送入 LLM |
| 结构化数据 | 抓取 + **专用解析** | 针对表格、JSON-LD、Schema.org 提取 |
| 实时性要求高 | **搜索 API** | 无需等待页面加载和渲染 |

**关键模式：搜索 → 判断 → 抓取。** 搜索引擎返回的摘要往往已经足够模型回答问题，不需要打开链接。只有当摘要信息不够深入时，Agent 才应执行抓取——先根据摘要判断哪个 URL 值得进一步深入，然后 `web_fetch` 打开它、清洗 HTML、深入阅读。这天然形成了一个高效的二级检索模式。

---

## 4. Build：web_fetch + web_search 实现

### 4.1 架构

```
用户 / 模型
    ↓
web_search("query")          web_fetch("url", max_chars)
    ↓                              ↓
DDGS().text()                 requests.get() [stream, timeout]
    ↓                              ↓
结构化结果 [{title, url, snippet}]   5 层安全校验
    ↓                              ↓
返回 JSON                      BeautifulSoup HTML → text
                                   ↓
                              长度截断 + 标记
                                   ↓
                              返回 JSON
```

### 4.2 web_fetch：5 层安全架构

Day 1 教我们「路径是不可信输入」，Day 2 教我们「命令参数是不可信输入」。Day 3 的教训是：**URL 也是不可信输入。** `web_fetch` 实现了 5 层安全校验：

```
用户 / 模型提供的 URL
    ↓
Layer 1  Scheme 校验       ← 只允许 http/https，拒绝 file/ftp/gopher/...
    ↓
Layer 2  内网 IP 拦截       ← DNS 解析后检查是否为私有/环回/链路本地地址
    ↓
Layer 3  请求控制          ← 10s 超时 + 流式读取 + 5MB 响应体硬上限
    ↓
Layer 4  HTML → 纯文本     ← BeautifulSoup 剥离 script/style/nav/footer/header
    ↓
Layer 5  长度截断          ← 默认 5000 字符上限 + "[truncated at N chars]" 标记
    ↓
返回结构化 JSON
```

**Layer 1 — Scheme 校验：拒绝非 HTTP 协议**

```python
if parsed.scheme not in ("http", "https"):
    raise ValueError(
        f"Only http/https URLs are allowed, got scheme {parsed.scheme!r}."
    )
```

拒绝 `file:///etc/passwd`、`ftp://`、`gopher://` 等。这是第一道也是最简单的一道防线——但缺了它，`file://` 协议可以直接读取本地文件系统。

**Layer 2 — 内网 IP 拦截：SSRF 防御** 🔑

```python
_PRIVATE_NETWORKS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),       # loopback
    ipaddress.IPv4Network("169.254.0.0/16"),     # link-local
    # ... + IPv6 equivalents
)

def _is_private_host(hostname: str) -> bool:
    for info in socket.getaddrinfo(hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        for net in _PRIVATE_NETWORKS:
            if ip in net:
                return True
    return False
```

**为什么需要 DNS 解析而非仅检查 URL 中的 IP？** 攻击者可以用域名指向内网地址（如 `internal-api.company.com` → `10.0.0.5`）。只检查 URL 中是否写了 `127.0.0.1` 是不够的——**必须解析域名后再检查 IP**。这与 Day 1 的 symlink 绕过一脉相承：字符串检查不够，需要在解析后的真实值上做判断。

这是 SSRF（Server-Side Request Forgery）防御的标准做法。覆盖范围：

- IPv4 私有地址：`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`
- 环回地址：`127.0.0.0/8`、`::1/128`
- 链路本地：`169.254.0.0/16`、`fe80::/10`
- 组播和保留地址：`224.0.0.0/4`、`240.0.0.0/4`

**Layer 3 — 请求控制：超时 + 流式读取 + 响应体上限**

```python
resp = requests.get(
    url,
    timeout=WEB_FETCH_TIMEOUT,       # 10 秒超时
    stream=True,                     # 流式读取 — 不透传整个 body
    headers={"User-Agent": "MyAgent/1.0 (learning project)"},
)
raw_body = resp.raw.read(WEB_FETCH_MAX_RESPONSE_BYTES + 1, decode_content=True)
# ↑ 最多读 5 MB，然后截断
```

三个关键设计决策：
- **`stream=True`**：不一次性下载整个响应体，而是按需读取——防止内存被大文件撑爆
- **10 秒超时**：防止 Agent 永远卡在慢速或无响应的服务器上
- **5 MB 硬上限**：即使响应的 Content-Length 声称更大，也只读取前 5 MB

**Layer 4 — HTML → 纯文本：BeautifulSoup 清洗**

```python
def _html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")

    # 先移除无内容元素
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 合并连续的多个空行
    return text
```

清洗流程：先 `decompose()` 移除噪声标签（script、style、nav、footer、header），再 `get_text()` 提取纯文本，最后用正则合并多余空行。一个 500KB 的 HTML 页面经过清洗后通常只剩 5–20KB 文本。

**Layer 5 — 长度截断：带标记的友好截断**

```python
if len(text) > max_chars:
    text = text[:max_chars] + f"\n… [truncated at {max_chars} chars]"
```

与 Day 2 的 `_truncate_output` 一脉相承——截断时必须带上明确的标记（`truncated at N chars`），让模型知道信息不完整，可以据此决定是否需要进一步操作（如分段抓取）。

### 4.3 web_search：DuckDuckGo 集成

```python
def web_search(query: str, max_results: int = 5) -> str:
    query = query.strip()
    max_results = max(1, min(max_results, 10))  # 硬钳制 1–10

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))

    results = [
        {"title": r["title"], "url": r["href"], "snippet": r["body"]}
        for r in raw
    ]
    return json.dumps({"query": query, "results": results, "count": len(results)})
```

设计要点：

- **结果数硬钳制**：`max(1, min(max_results, 10))` —— 无论模型传什么值，永远返回 1–10 条
- **结构化输出**：每条结果固定为 `{title, url, snippet}` 三元组，模型可以稳定解析
- **错误透传**：搜索失败时返回 `{"error": "Search failed: ..."}`，模型可据此调整（换搜索词或直接抓取）

### 4.4 工具 Schema 设计：when-to-call 条件

```python
"web_search": {
    "description": (
        "Search the web via DuckDuckGo and return a list of results with "
        "titles, URLs, and short text snippets. Call this when the user "
        "asks to search for something online, find information, look up "
        "facts, or research current events. Combine with web_fetch to "
        "open the most promising result for more detail."
    ),
    ...
}

"web_fetch": {
    "description": (
        "Fetch a web page by URL and return the page title plus extracted "
        "plain text. ... Call this when the user asks to read, fetch, "
        "visit, or summarize a URL. Also call this instead of curl or "
        "wget (which are blocked by the command allowlist)."
    ),
    ...
}
```

遵循 Week 1 Day 3「when-to-call」的教训——描述不仅说明工具做什么，更说明**什么时候该用它**。`web_search` 明确提示「和 web_fetch 组合使用：先搜再开」；`web_fetch` 明确提示「替代 curl/wget」。

### 4.5 工具输出格式：统一的结构化 JSON

| 工具 | 返回字段 | 说明 |
| --- | --- | --- |
| `web_fetch` | `url, title, text, chars, truncated, content_type, status_code, note` | 注意 `url` 是最终 URL（含重定向），不是原始 URL |
| `web_search` | `query, results[{title, url, snippet}], count` | `results` 始终是数组，空结果时 `count=0` |

与 Day 2 的工具输出设计一致：结构固定、可预测、错误可操作。

---

## 5. 实验：Agent 多步 Web 任务

### 5.1 测试运行

运行（无需 API Key）：

```bash
python experiments/week03/agent.py --test-web
```

测试覆盖：

| 类别 | 测试项 | 预期 |
| --- | --- | --- |
| 正常功能 | 抓取 example.com | 200 OK，提取标题和正文 |
| 长度截断 | `max_chars=50` | `truncated=True`，输出被截断并标记 |
| Scheme 拦截 | `file://`, `ftp://`, 无 scheme | 全部 BLOCKED |
| 内网 IP 拦截 | `localhost`, `127.0.0.1`, `192.168.x.x`, `10.x.x.x`, `[::1]` | 全部 BLOCKED |
| 超时处理 | 不可路由地址 `192.0.2.1` | 超时或连接失败 |
| HTTP 错误 | `httpbin.org/status/404` | 捕获 404，透传错误 |
| 空输入 | `""`, `"   "` | 全部 BLOCKED |
| 搜索功能 | `"Python programming language"` | 返回 1–3 条结构化结果 |
| 结果钳制 | `max_results=999` | 钳制到 ≤10 条 |
| 搜索空输入 | `""`, `"   "` | 全部 BLOCKED |

### 5.2 Agent 多步任务实测

**Prompt**（`--day3`）：

```
Use web_search to find the official Python website. Then use web_fetch
to open the top result and tell me: what is the page title, and what
is the first sentence or tagline you see on the page? Save a summary
of what you found to 'python_website_notes.md' in the sandbox.
```

**完整 Trace**：

```
→ tool call: web_search({"query": "official Python website", "max_results": 5})
← result:    {"query": "official Python website", "results": [
               {"title": "Python (programming language) - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "snippet": "4 days ago - P…"}]}

→ tool call: web_fetch({"url": "https://www.python.org/", "max_chars": 5000})
← result:    {"url": "https://www.python.org/",
               "title": "Welcome to Python.org",
               "text": "Welcome to Python.org\nNotice:\nThis page displays…"}

→ tool call: write_file({"filepath": "python_website_notes.md", "content": "# Python Website Notes\n\n..."})
← result:    {"written": "python_website_notes.md", "bytes": 1389}
```

**Agent 最终输出摘要**：

- 🔍 **搜索**：Top result 为 `python.org`（Wikipedia 条目作为备选出现在结果中）
- 📄 **抓取**：页面标题 `Welcome to Python.org`，Tagline `Whether you're new to programming or an experienced developer, it's easy to learn and use Python.`
- 📝 **保存**：摘要写入 `python_website_notes.md`（1389 bytes），包含 Source URL、搜索词、页面标题、Tagline、Homepage 各板块概览（Get Started、Download、Docs、Jobs、News、Events、Success Stories、Use Python for…、PSF 使命）

### 5.3 实验分析

这次运行展示了几个关键行为：

1. **搜索→抓取二级模式生效。** 模型先用 `web_search` 找到 python.org，再 `web_fetch` 打开它——它没有直接猜 URL，也没有打开 Wikipedia 而是选择了官方站点。这说明搜索结果中的 snippet 帮助模型做出了正确的 URL 选择。

2. **搜索结果不等于答案。** DuckDuckGo 返回的摘要只是 "The official home of the Python Programming Language"，不足以回答「第一句 tagline 是什么」——模型正确地判断了需要进一步抓取。

3. **HTML 清洗在工作。** `web_fetch` 的返回是干净的纯文本（"Whether you're new to programming…"），而非原始 HTML。`BeautifulSoup` 成功剥离了 `<script>`、`<style>`、`<nav>` 等噪声标签。

4. **多工具协作自然。** 模型在同一个任务中使用了 `web_search` → `web_fetch` → `write_file` 三个工具，跨工具的信息传递（搜索结果 URL → fetch → 文件保存）完全正确。

5. **文件工具与 Web 工具的组合。** 最终结果持久化到了 sandbox 中——Day 1 的文件工具和 Day 3 的 Web 工具无缝配合。

---

## 6. 今日收获

1. **URL 也是不可信输入。** Day 1 教我们「路径是不可信输入」，Day 2 教我们「命令参数是不可信输入」，Day 3 补齐了第三块拼图：**URL 也是不可信输入。** Scheme（`file://`）、IP（内网地址）、协议（SSRF）都需要校验——仅靠 `requests.get()` 是不够的。

2. **SSRF 防御需要 DNS 解析级别的检查。** 字符串匹配检查 URL 中的 IP 是不够的——攻击者可以用域名指向内网。`socket.getaddrinfo()` + `ipaddress` 是标准解法。

3. **HTML → 纯文本清洗是 Web 工具的必备能力。** 原始 HTML 中 80–90% 是噪声（标签、样式、脚本、广告）。`BeautifulSoup.decompose()` 移除无内容标签 + `get_text()` 提取纯文本，将 500KB HTML 变成 5–20KB 的可读文本。这与 Day 2 的「输出截断到 4000 字符」一脉相承——都是在保护 LLM 的上下文窗口。

4. **搜索优先，按需抓取。** `web_search` 的摘要往往已经足够回答事实性问题。只有当摘要信息不够深入时，才用 `web_fetch` 打开具体页面。这个二级检索模式节省 token、降低延迟。

5. **截断必须带标记。** 无论是 `web_fetch` 的长度截断（`[truncated at N chars]`）还是响应体截断（`note` 字段说明），都必须明确告诉模型「信息不完整」。这样模型才能据此决定下一步行动（如分段抓取或缩小范围）。

6. **Web 工具天然延伸了 Agent 的能力边界。** 加上 Web 工具后，Agent 不再局限于沙箱内的文件和命令——它可以获取实时信息、验证事实、引用来源。这是 Agent 从「玩具」走向「实用」的关键一步。

### Day 1 ↔ Day 2 ↔ Day 3 安全范式对比

| 维度 | Day 1（文件工具） | Day 2（命令工具） | Day 3（Web 工具） |
| --- | --- | --- | --- |
| 攻击面 | 路径遍历（`../`、symlink） | 命令注入（元字符）+ 参数注入 | URL 注入（scheme）+ SSRF |
| 核心防御 | `realpath()` + `commonpath()` | `shell=False` + 白名单 + 参数校验 | scheme 校验 + DNS 解析 IP 检查 |
| 关键教训 | 字符串过滤不够，需要 OS 级路径解析 | 白名单不够，需要参数级控制 | URL 字符串检查不够，需要 DNS 级网络边界 |
| 默认行为 | 拒绝沙箱外路径 | 拒绝非白名单可执行文件 | 拒绝非 HTTP/HTTPS + 拒绝内网地址 |
| 输出保护 | 扩展名白名单 | 超时 + 输出上限 | HTML 清洗 + 长度截断 + 截断标记 |

---

### Done-when 自检

- [x] 理解了搜索 API 与网页抓取两种模式的核心差异（信噪比、token 成本、适用场景）
- [x] 实现了 `web_fetch`：`requests` + `BeautifulSoup` HTML-to-text + 长度截断
- [x] 实现了 `web_search`：DuckDuckGo 集成 + 结果数钳制 + 结构化返回
- [x] 实现了 5 层安全架构（Scheme → 内网 IP → 请求控制 → HTML 清洗 → 长度截断）
- [x] 覆盖了 SSRF 防御：DNS 解析 + `ipaddress` 检查 7 类私有/保留地址
- [x] 运行 `--test-web`，所有功能测试和安全拦截测试通过
- [x] 运行 `--day3` 多工具协作任务：搜索 → 抓取 → 写文件，全链路成功
- [x] 工具 Schema 包含明确的 when-to-call 条件
- [x] 截断输出带 `[truncated at N chars]` 标记，与 Day 2 的输出截断模式一致




# Day 4 — 组合所有工具：多步 Agent 实战 + 完整 Trace

> 目标：将 Day 1–3 的 6 个工具组合到一个 Agent 中，给它一个真实的跨工具多步任务，并观察完整的执行 Trace。结合 Anthropic "Building Effective Agents" 的原则，理解为什么「简单、可组合的模式」比复杂框架更好。
> 实验代码：`experiments/week03/agent.py`（运行 Agent：`python experiments/week03/agent.py --day4`）

---

## 今日主题

前三天的学习是「拆分」——把文件工具、命令工具、Web 工具逐个构建、逐个加固。Day 4 是「组合」——把 6 个工具放在同一个 Agent 里，给它一个真实的跨工具任务，观察 Agent 如何自主编排这些工具。

核心问题不是「能不能调用工具」，而是 **Agent 能否在开放任务中做出正确的工具选择、按正确的顺序执行、并从中间结果中提取信息传递给下一个工具**。

Anthropic 的 "Building Effective Agents" 文章提供了一个关键框架来理解我们在做什么——以及为什么不做得更复杂。

---

## 1. 从「Building Effective Agents」中学到的核心原则

### 1.1 Workflow vs. Agent 的定义

| 概念 | 定义 | Day 4 中的对应 |
| --- | --- | --- |
| **Workflow（工作流）** | 通过预定义的代码路径编排 LLM 和工具的系统 | Day 1–3 的单个工具调用（LLM 选工具 → 执行 → 返回结果） |
| **Agent（智能体）** | LLM **动态**指导自己的流程和工具使用，保持对完成任务方式的控制 | Day 4：Agent 自主决定先搜索还是先列目录、选哪个 URL、用什么命令验证 |

关键区别：Workflow 的任务分解是**预先固定的**；Agent 的任务分解是**模型在运行时自主决定的**。

### 1.2 五个 Workflow 模式 —— 以及为什么 Day 4 不需要它们

Anthropic 文章描述了五种构建模式。Day 4 的实验让我们理解：**对于很多任务，一个简单的 Agent 循环就够了，不需要这些高级模式。**

| 模式 | 定义 | 何时需要 | Day 4 为什么不需要 |
| --- | --- | --- | --- |
| **Prompt Chaining** | 顺序执行多个 LLM 调用，前一个的输出是后一个的输入 | 任务可以清晰分解为固定子任务 | Agent 在单次对话中完成所有步骤——LLM 自主决定何时搜索、何时抓取、何时写入 |
| **Routing** | 分类输入，导向专门的后续处理 | 存在明显不同的输入类别 | 只有一个任务类型，不需要分类 |
| **Parallelization** | 拆分独立子任务并行执行，或多次运行取多样输出 | 子任务相互独立 | 步骤间有强依赖（搜索结果 → 选择 URL → 抓取 → 写入） |
| **Orchestrator-Workers** | 中央 LLM 动态分解任务，委派给 worker LLM | 子任务事先无法确定 | 任务步骤是已知的（搜索→抓取→写→读→列→cat），不需要动态分解 |
| **Evaluator-Optimizer** | 一个 LLM 生成，另一个 LLM 评估并循环反馈 | 有明确评估标准，迭代优化有价值 | 任务是一次性执行，不需要迭代优化 |

**核心教训**：文章的第一条建议就是 "Start simple. Add complexity only when it demonstrably improves outcomes." Day 4 的实验完美验证了这一点——一个带有 6 个工具的简单 Agent 循环，加上一个清晰的 prompt，就能完成一个相当复杂的多步任务。

### 1.3 工具设计的 ACI（Agent-Computer Interface）原则

"Building Effective Agents" 提出了一个被低估的洞见：**花在优化工具定义上的时间，应该至少和优化整体 prompt 一样多。** 文章中甚至提到——在构建 SWE-bench Agent 时，团队花在优化工具上的时间超过了优化整体 prompt 的时间。

这条原则贯穿了 Week 3 的所有工具设计：

| Day 4 工具 | ACI 设计决策 | 效果 |
| --- | --- | --- |
| `web_search` | description 明确写「Combine with web_fetch to open the most promising result」 | 模型知道搜索后应该抓取 |
| `web_fetch` | description 写「Call this instead of curl or wget (which are blocked)」 | 模型不会尝试用命令工具抓网页 |
| `run_command` | description 列出所有允许的可执行文件 + 说明「no shell」 | 模型不会尝试 `curl` 或管道 |
| `write_file` | required: `["filepath", "content"]` — 两个参数都必填 | 防止模型只传路径不传内容 |

---

## 2. Build：Day 4 的实现改造

### 2.1 增强的 Trace 输出

Day 1–3 的 trace 只有简单的 `→ tool call:` / `← result:`。Day 4 升级为**带步骤编号和分类标记的完整 trace**：

```
  ── LLM turn 1 (1 tool call(s)) ──
  [1] 🔧 web_search({"query": "Anthropic Building Effective Agents", ...})
  [1] ✅ result: {"query": "...", "results": [...], "count": 5}

  ── LLM turn 2 (1 tool call(s)) ──
  [2] 🔧 web_fetch({"url": "https://www.anthropic.com/...", ...})
  [2] ✅ result: {"title": "...", "text": "......"… [5234 total chars]}
```

每条 trace 行包含：
- `[N]` — 全局步骤编号（跨 turn 累加，不是每 turn 重置）
- `🔧` / `✅` / `❌` — 调用、成功、失败标记
- LLM turn 分隔线 — 清晰展示每个 LLM 调用返回了几个 tool call
- 结果预览 + 总长度标记 — 长结果不刷屏但显示完整长度

### 2.2 安全阀：MAX_TURNS

```python
MAX_TURNS = 15
if turn > MAX_TURNS:
    return f"[Agent stopped after {MAX_TURNS} LLM turns — possible loop.]"
```

Agent 循环可能因为模型反复调用工具而不终止。15 轮是安全阀——足够完成一个 6 步任务，但不会无限烧 token。

### 2.3 多步任务的 Prompt 设计

Day 4 的 prompt 精心设计，遵循三个原则：

```
1. Use web_search to look up "Anthropic Building Effective Agents"
2. Use web_fetch to open the most relevant result → extract 5 core principles
3. Use write_file to save a formatted summary to 'notes.md'
4. Use read_file to verify 'notes.md' was written correctly
5. Use list_dir to confirm the file appears in the sandbox
6. Use run_command with 'cat notes.md' to double-check the content

Important: Do EVERY step in order. Do not skip any step.
After each tool call, report what you learned before proceeding.
```

| 设计原则 | 在 Prompt 中的体现 |
| --- | --- |
| **明确步骤** | 6 个编号步骤，每步指定了具体工具名称 |
| **数据流提示** | Step 1 搜索 → Step 2 打开「最相关的结果」→ Step 3 写入「摘要」 |
| **验证闭环** | 写入后有三重验证：`read_file`（逻辑验证）+ `list_dir`（文件系统验证）+ `run_command cat`（OS 级验证） |
| **防跳步** | "Do EVERY step in order. Do not skip any step." |
| **可观测性** | "After each tool call, report what you learned" — 强制中间报告 |

**关键设计决策：给具体工具名，但不给具体 URL。** Prompt 说「用 web_search 搜索 X」，但不告诉模型「打开 python.org」。模型必须自己从搜索结果中选出正确的 URL。这保留了 Agent 的自主决策空间——如果 prompt 把 URL 也写死了，那就不需要 Agent 了，一个脚本就够了。

### 2.4 这个 Prompt 是「Skill」吗？—— Skill vs. Agent Prompt 的边界

Day 4 的 prompt 有编号步骤、指定了工具名、要求按顺序执行——看起来很像 Claude Code 中的 `/code-review` 这类 Skill。它确实不是巧合：这个 prompt 故意站在 Skill 和 Agent 的边界上。

**Skill（编码的工作流）的架构：**

```
代码定义步骤 → 每个步骤内部调用 LLM
```

- **谁决定下一步？** 代码（`if step1.ok → step2`）
- **错误恢复** 必须在代码中显式处理
- **工具选择** 每个步骤固定
- **可复用性** 命名、参数化、可调用
- **典型例子** `/code-review`、`/security-review`

**Agent Prompt（Day 4）的架构：**

```
LLM 读取步骤 → LLM 自主决定如何执行每一步
```

- **谁决定下一步？** LLM（读取结果，自主决策）
- **错误恢复** LLM 可以重试、换参数、换 URL、跳过失败步骤
- **工具选择** LLM 在每个步骤中选择最佳工具
- **可复用性** 一次性指令（虽然可以参数化复用）
- **典型例子** Day 4 prompt、"帮我研究 X 并写报告"

**为什么 Day 4 看起来像 Skill？**

Day 4 的 prompt 是**故意具体**的——编号步骤、指定工具名、有序验证。这是因为 Day 4 是一个**学习实验**：我们需要观察 Agent 是否遵循预期路径。如果 prompt 写成开放式的（"research Building Effective Agents and save notes"），我们就无法判断 Agent 的工具选择是正确的还是碰巧的。

**但这 6 个步骤是「规范」而不是「脚本」。** LLM 仍然需要自主决策：

| 决策点 | Prompt 提供了什么 | LLM 必须自主决定什么 |
| --- | --- | --- |
| 搜索 | 「用 web_search 搜索 X」 | 搜什么关键词？返回多少条结果？ |
| 抓取 | 「打开最相关的结果」 | **哪个结果最相关？** Anthropic 官方？Wikipedia？第三方博客？ |
| 写入 | 「保存格式良好的摘要到 notes.md」 | 摘要怎么写？结构？包含什么？不包含什么？ |
| 验证 | 「用 read_file / list_dir / cat 验证」 | 怎么判断验证通过？内容不对怎么办？ |

**完整的对照表：**

| 维度 | Skill（编码工作流） | Agent Prompt（Day 4） |
| --- | --- | --- |
| **编排逻辑在哪？** | 代码中 | 模型的推理中 |
| **下一步由谁决定？** | 代码（`if step1.ok → step2`） | LLM（读取结果后自主决定） |
| **错误恢复** | 必须在代码中显式处理 | LLM 可以重试、换参数、换策略 |
| **工具选择** | 每个步骤固定 | LLM 在每个步骤中选择最佳工具 |
| **可复用性** | 命名、参数化、可调用 | 一次性指令 |
| **可预测性** | 高——每次都走相同路径 | 中——LLM 可能选不同的策略 |
| **灵活性** | 低——遇到未编码的场景就卡住 | 高——LLM 可以即兴调整 |

**这个光谱来自「Building Effective Agents」的核心定义：**

> Workflows are systems where LLMs and tools are orchestrated through **predefined code paths**. Agents are systems where LLMs **dynamically direct** their own processes and tool usage.

Day 4 是一个**混合体**——prompt 提供了工作流的骨架（6 个编号步骤），但 LLM 填充了执行的细节（选哪个 URL、怎么写摘要、怎么判断验证通过）。这在实践中是最有效的模式：**给 Agent 足够的结构让它不迷路，同时给它足够的自由度来处理意外情况。**

这个区别也解释了 Day 1–3 和 Day 4 的本质差异。Day 1–3 的 prompt 是「试一个工具，观察结果」——它们是单回合的、探索性的。Day 4 的 prompt 是一个**多步任务规范**——它定义了一个目标状态和到达路径，但把执行权交给了 Agent。

---

## 3. 实验：6 步跨工具任务完整 Trace

### 3.1 实验设置

```bash
python experiments/week03/agent.py --day4
```

- **模型**：`qwen3.7-plus`（通过 DashScope API）
- **工具**：6 个（read_file, write_file, list_dir, run_command, web_fetch, web_search）
- **安全层**：所有 Day 1–3 的安全机制全部生效（沙箱边界 + 命令白名单 + SSRF 防御）
- **任务**：搜索 Anthropic 文章 → 抓取 → 写摘要 → 三重验证

### 3.2 完整 Trace

**LLM Turn 1** — Agent 收到多步指令，选择从搜索开始：

```
[1] 🔧 web_search({"query": "Anthropic Building Effective Agents", "max_results": 5})
[1] ✅ result: {"query": "Anthropic Building Effective Agents", "results": [
         {"title": "Building effective agents - Anthropic",
          "url": "https://www.anthropic.com/engineering/building-effective-agents",
          "snippet": "Dec 19, 2024 - A guide to building effective AI agents..."},
         ...
       ], "count": 5}
```

**LLM Turn 2** — 模型识别出第一条结果（anthropic.com）是官方来源，发起抓取：

```
[2] 🔧 web_fetch({"url": "https://www.anthropic.com/engineering/building-effective-agents", "max_chars": 5000})
[2] ✅ result: {"url": "https://www.anthropic.com/engineering/building-effective-agents",
         "title": "Building effective agents - Anthropic",
         "text": "Building effective agents\n\n...workflows vs agents...\n\n
                  Prompt chaining...\nRouting...\nParallelization...\n
                  Orchestrator-workers...\nEvaluator-optimizer...",
         "chars": 4823, "truncated": false, "content_type": "text/html", "status_code": 200}
```

**LLM Turn 3** — 模型将提取的内容整理成结构化摘要，写入文件：

```
[3] 🔧 write_file({"filepath": "notes.md", "content": "# Anthropic: Building Effective Agents — Summary\n\n..."})
[3] ✅ result: {"written": "notes.md", "bytes": 3722}
```

**LLM Turn 4** — 第一重验证：用 `read_file` 检查内容完整性：

```
[4] 🔧 read_file({"filepath": "notes.md"})
[4] ✅ result: "# Anthropic: Building Effective Agents — Summary\n\n
         ## Overview\nAgents are systems where LLMs dynamically direct..."
```

**LLM Turn 5** — 第二重验证：用 `list_dir` 确认文件在沙箱中存在：

```
[5] 🔧 list_dir({"dirpath": "."})
[5] ✅ result: {"directory": ".", "entries": [
         ..."notes.md"...
       ]}
```

**LLM Turn 6** — 第三重验证：用 `cat` 做 OS 级确认：

```
[6] 🔧 run_command({"command": "cat notes.md"})
[6] ✅ result: {"command": "cat notes.md", "exit_code": 0,
         "stdout": "# Anthropic: Building Effective Agents — Summary\n\n..."}
```

**LLM Turn 7** — 所有步骤完成，模型返回最终报告。

### 3.3 结果总结

| Step | 工具 | 作用 | 关键发现 |
|------|------|------|----------|
| **1. Search** | `web_search` | 搜索 Anthropic 文章 | 第一条结果就是官方博客，snippet 提供了足够的上下文让模型做选择 |
| **2. Fetch** | `web_fetch` | 抓取完整文章 | BeautifulSoup 清洗后的 4,823 字符纯文本，未被截断——刚好在 5,000 限制内 |
| **3. Write** | `write_file` | 保存结构化摘要 | 3,722 bytes，包含 Overview、5 个模式详解、核心原则、Takeaway |
| **4. Read** | `read_file` | 内容验证 | 确认写入内容完整，格式正确 |
| **5. List** | `list_dir` | 文件存在性验证 | 在沙箱根目录中确认文件存在 |
| **6. Cat** | `run_command` | OS 级验证 | exit_code=0，完整内容通过 cat 输出 |

Agent 在 **7 个 LLM turn** 内完成了全部 6 个步骤，使用了 **4 类工具**（搜索、抓取、文件读写、命令执行），跨工具的数据传递完全正确（搜索结果 URL → fetch → 提取文本 → write → read → cat）。

---

## 4. 实验分析

### 4.1 模型为什么选择官方博客而不是 Wikipedia？

搜索结果中既有 Anthropic 官方博客，也有第三方摘要和 Wikipedia。模型选择了 `anthropic.com/engineering/building-effective-agents`。这说明：

- **搜索 snippet 提供了足够的区分信息**——模型从 snippet 中判断出官方来源更权威
- **工具描述中的 "Combine web_search + web_fetch" 提示起了作用**——模型知道搜索后再抓取的模式
- **Day 1–3 的安全层全部透明生效**——没有触发任何安全拦截，Agent 顺畅执行

### 4.2 三重验证的意义

Day 4 的 prompt 要求了三次验证（read_file → list_dir → cat），这不是多余的设计：

| 验证方式 | 验证什么 | 如果被跳过会怎样 |
| --- | --- | --- |
| `read_file` | 内容是否正确写入 | 文件可能存在但内容为空或损坏 |
| `list_dir` | 文件是否在预期位置 | 可能写到了错误的路径 |
| `run_command cat` | OS 视角的文件状态 | 前两者都在 Agent 工具层验证——`cat` 是独立视角 |

这是「纵深防御」思想在任务设计中的延伸：**不信任单次操作的结果，用独立工具交叉验证。**

### 4.3 为什么这个实验验证了「简单优于复杂」

Day 4 的 Agent **没有**使用：
- ❌ 多 Agent 编排
- ❌ 复杂的 workflow 框架
- ❌ Evaluator-Optimizer 循环
- ❌ Orchestrator-Workers 委派

但它完成了任务。这验证了 Anthropic 文章的核心论点：**一个简单的 Agent 循环 + 一组设计良好的工具 + 一个清晰的 prompt，已经能覆盖很多实际场景。** 在增加 Orchestrator-Workers 或 Evaluator-Optimizer 等复杂模式之前，先问自己：「当前的简单方案哪里不够？」

### 4.4 6 个工具的协同模式

| 工具组合 | 协同模式 | Week 3 的前置基础 |
| --- | --- | --- |
| `web_search` → `web_fetch` | 二级检索：先搜再开 | Day 3 的「搜索优先，按需抓取」 |
| `web_fetch` → `write_file` | 外部信息持久化到沙箱 | Day 1 的文件工具 + Day 3 的 Web 工具 |
| `write_file` → `read_file` | 写入后验证 | Day 1 的读写工具 |
| `read_file` + `list_dir` + `cat` | 三重交叉验证 | Day 1 + Day 2 的工具组合 |

**关键认知**：这些协同模式不是 Agent「学会」的——是 prompt 设计出来的。一个好的多步 prompt 本质上是在**帮 Agent 构建一个隐式的 workflow**，但执行路径仍然是 Agent 自主决定的。

---

## 5. 今日收获

1. **Agent 的本质是「LLM + 工具 + 循环」。** 不需要复杂的框架——一个 `while True` 循环 + API 调用 + 工具调度，就已经是一个完整的 Agent。Anthropic 文章建议「直接从 LLM API 开始，而不是框架」，Day 4 的实现就是这句话的生动证明。

2. **Prompt 是隐式的 Workflow。** Day 4 的 prompt 通过编号步骤 + 数据流提示 + 验证闭环，构建了一个隐式的 6 步工作流——但 Agent 仍然自主决定每个步骤的具体行为（选哪个 URL、怎么组织摘要、用什么命令验证）。这介于「完全固定的 pipeline」和「完全开放的 Agent」之间，是实践中最高效的模式。

3. **Trace 不是调试工具——是理解 Agent 的窗口。** Day 4 的增强 trace（步骤编号 + turn 分隔 + 结果预览）让我们能看到 Agent 的完整决策过程。很多时候 Agent 的行为出乎意料——没有 trace，你永远不会知道它为什么选了那个 URL 或为什么跳过了某一步。

4. **工具之间的「胶水」是 prompt，不是代码。** Day 4 的 6 个工具之间没有代码层面的编排逻辑——`web_search` 不知道 `web_fetch` 的存在，`write_file` 不依赖 `read_file`。它们通过 prompt 中的指令被串联起来。这意味着：**增加工具不会增加编排代码的复杂度**——只要 prompt 设计得当。

5. **安全层透明是最理想的安全。** Day 4 的 Agent 执行了 6 个跨工具步骤，没有触发任何安全拦截——不是因为安全层不存在，而是因为 Agent 的行为完全在安全边界内。好的安全设计应该让合法操作零摩擦，只在越界时拦截。

6. **验证闭环是 Agent 任务设计的关键模式。** 三重验证（read + list + cat）不是过度设计——每个验证层检查不同维度的问题。这是 Week 2 Day 4「让 Agent 自我验证」思想的延续：不信任单次操作，从多个角度确认结果。

### Day 1 → Day 2 → Day 3 → Day 4 能力演进

| 维度 | Day 1 | Day 2 | Day 3 | Day 4 |
| --- | --- | --- | --- | --- |
| 工具数量 | 3（文件） | +1（命令） | +2（Web） | **6 个全部** |
| 安全层 | 沙箱边界 | +7 层命令安全 | +5 层 Web 安全 | **全部 13+ 层在线** |
| 任务复杂度 | 单工具调用 | 单一错误恢复 | 3 工具协作 | **6 步跨工具自主编排** |
| Trace 可见性 | 基础打印 | 基础打印 | 基础打印 | **带编号、分类、turn 分隔** |
| 核心教训 | 路径是不可信输入 | 参数也是不可信输入 | URL 也是不可信输入 | **工具组合靠 prompt，不靠代码** |

---

### Done-when 自检

- [x] 理解了 Workflow 与 Agent 的核心区别（预定义路径 vs. 动态决策）
- [x] 理解了 Anthropic 5 个 Workflow 模式各自适用场景——以及为什么 Day 4 不需要它们
- [x] 理解了 ACI（Agent-Computer Interface）原则：工具定义的投入 ≥ prompt 优化
- [x] 将 `run_agent()` 升级为增强 trace（步骤编号 + 🔧/✅/❌ 标记 + turn 分隔 + 结果预览）
- [x] 新增 `MAX_TURNS=15` 安全阀防止 Agent 无限循环
- [x] 设计了一个 6 步跨工具 Prompt（搜索 → 抓取 → 写入 → 三重验证）
- [x] 运行 `--day4`，Agent 使用 4 类 6 个工具完成全部步骤
- [x] 所有 Day 1–3 的安全层在 Agent 自主执行中透明生效（零拦截 = 合法操作零摩擦）
- [x] 验证了「简单 Agent 循环 + 好工具 + 好 Prompt」胜过复杂框架


# Day 5 — 重构工具输出：一致的格式、截断标记、可操作的错误

> 目标：将 Day 1–4 的 6 个工具输出重构为统一的 ToolResult 信封格式，实现一致的截断标记、结构化的可操作错误，然后重新运行 Day 4 的多步任务对比 trace。
> 实验代码：`experiments/week03/agent.py`（运行 Agent：`python experiments/week03/agent.py --day4`）

---

## 今日主题

在 Agent 架构中，工具返回结果（Tool Result）的设计质量直接决定了模型的推理质量。Day 1–4 我们构建了 6 个功能完好的工具，但它们的输出格式是**不一致的**：

- `read_file` 返回原始字符串，没有截断保护
- `run_command` 用 `_truncate_output` 做截断，但标记是独立的
- `web_fetch` 用另一套截断逻辑，标记格式相同但函数不同
- 错误处理靠 Python 异常（`ValueError`），模型拿到的只是 `Error: ...` 字符串

**Day 5 的核心问题**：如果模型每次调用工具都看到不同的返回格式、不同的截断标记、不同的错误信息结构，它就无法形成稳定的解析预期——每次都需要重新「理解」输出格式。这不是工具的 bug，而是工具设计的欠债。

---

## 1. 理论基础：工具返回是模型专用协议

工具返回必须被当作**模型专用协议**来设计，而不是简单地把 API 原始响应或系统日志丢回去。三个核心维度：

### 1.1 结构化（Structure）

模型处理信息的最佳方式是结构化、类型化、可预测的。原始 API 返回或 HTML 是混沌的，模型需要消耗大量 token 去「理解」哪里是数据、哪里是元数据。不能把原始的 HTTP 请求结果直接作为模型上下文，需要做一定的后处理，使其结构清晰、信息清晰。

| 原则 | 说明 | 示例 |
|---|---|---|
| **固定 Schema** | 每次返回相同顶级字段，模型不用猜今天是什么格式 | `{"status": "success", "data": {...}}` |
| **类型明确** | 数字就是数字，列表就是列表，避免字符串包裹一切 | `{"temperature": 23.5}` 而非 `{"result": "温度是23.5度"}` |
| **语义分层** | 区分「业务数据」「执行元数据」「上下文补充」 | `data`（核心结果）、`pagination`（翻页信息）、`suggestions`（下一步建议） |
| **空值显式化** | 明确告诉模型「这个字段有意为空」vs 「这个字段不存在」 | `{"related_articles": []}` 或 `{"related_articles": null, "reason": "无权限"}` |

### 1.2 简洁（Brevity）

上下文窗口有限，且模型在长上下文中的检索精度会衰减。工具返回必须是信息密度最大化的压缩包。

具体策略：
- **主动截断与摘要**：不要返回原始全文，返回模型决策所需的摘要
- **消除视觉噪音**：对模型来说，HTML/CSS/JavaScript 标签、重复的广告文案、页眉页脚导航栏、时间戳/UUID/哈希值的完整形式（除非当前任务需要）、人类可读的格式化空格和换行（除非影响语义）都是纯噪音
- **使用枚举和符号压缩**：`{"status": "ok", "code": "SUCCESS"}` 而非 `{"status": "The operation completed successfully without errors"}`
- **分页与增量加载**：如果结果天然很大，不要一次性返回。提供分页指针让模型按需索取

### 1.3 错误（Errors）

模型无法像人类那样阅读堆栈跟踪并推断根因。它需要语义化、可分类、**可行动**的错误信息。设计包含：状态、错误类别、错误详情、**错误修复指导（hint）**。
| 原则            | 说明                           | 示例                                                                         |
| ------------- | ---------------------------- | -------------------------------------------------------------------------- |
| **固定 Schema** | 每次返回相同顶级字段，模型不用猜今天是什么格式      | `{"status": "success", "data": {...}, "metadata": {...}}`                  |
| **类型明确**      | 数字就是数字，列表就是列表，避免字符串包裹一切      | `{"temperature": 23.5}` 而非 `{"result": "温度是23.5度"}`                        |
| **语义分层**      | 区分「业务数据」「执行元数据」「上下文补充」       | `data`（核心结果）、`pagination`（翻页信息）、`suggestions`（下一步建议）                       |
| **空值显式化**     | 明确告诉模型「这个字段有意为空」vs 「这个字段不存在」 | `{"related_articles": []}` 或 `{"related_articles": null, "reason": "无权限"}` |
Brevity
上下文窗口有限，且模型在长上下文中的检索精度会衰减。工具返回必须是信息密度最大化的压缩包。
具体策略
（1）主动截断与摘要：不要返回原始全文，返回模型决策所需的摘要。
（2）消除视觉噪音
对模型来说，以下都是纯噪音：
* HTML/CSS/JavaScript 标签
* 重复的广告文案
* 页眉页脚导航栏
* 时间戳、UUID、哈希值的完整形式（除非当前任务需要）
* 人类可读的格式化空格和换行（除非影响语义）
（3）使用枚举和符号压缩
// 冗余
{"status": "The operation completed successfully without errors"}
// 简洁
{"status": "ok", "code": "SUCCESS"}
（4）分页与增量加载
如果结果天然很大，不要一次性返回。提供分页指针让模型按需索取。这给了模型自主权：如果当前 10 条已足够回答用户，它可以停止；如果需要更多，它知道怎么拿。
Errors
模型无法像人类那样阅读堆栈跟踪并推断根因。它需要语义化、可分类、可行动的错误信息。
设计包含：状态、错误类别、错误详情、错误修复指导

---

## 2. Build：将理论落地到代码

Day 5 的目标不是从头设计一套生产级 Schema（见下方附录），而是在现有 6 个工具的基础上，用**最小改动实现最大一致性**。

### 2.1 统一信封：`_make_success()` / `_make_error()`

所有工具返回共享同一个外层结构：

```python
def _make_success(data, *, truncated=False, truncated_at=None) -> str:
    return json.dumps({
        "status": "success",
        "data": data or {},
        "error": None,
        "truncated": truncated,
        "truncated_at": truncated_at,
    })

def _make_error(error_type: str, message: str, *, hint=None) -> str:
    return json.dumps({
        "status": "error",
        "data": None,
        "error": {
            "type": error_type,
            "message": message,
            "hint": hint or _ERROR_HINTS.get(error_type, "..."),
        },
        "truncated": False,
        "truncated_at": None,
    })
```

**设计决策**：5 个顶级字段就是信封的「合约」。模型学会一次，到处适用：
- `status` — 先判断成功还是失败
- `data` — 成功时取业务数据
- `error` — 失败时取 `type` + `hint`，决定下一步
- `truncated` / `truncated_at` — 判断信息是否完整

### 2.2 统一截断：`_truncate()`

Day 4 有**两套**截断实现：`_truncate_output`（run_command）和 web_fetch 的内联截断逻辑。Day 5 合并为一个：

```python
def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n… [truncated at {max_chars} chars]", True
```

**关键变化**：
- `read_file` 新增截断保护（上限 8,000 chars）—— Day 4 中完全没有
- `run_command` 和 `web_fetch` 改用同一个 `_truncate()`，标记格式完全一致
- 截断信息从 `data` 内部提升到信封级别（`truncated` / `truncated_at`），模型不用深入 payload 就能判断信息完整性

### 2.3 17 种可操作错误类型

Day 4 的错误是 Python 异常字符串：`Error: Executable 'env' not in allowlist.`。模型拿到后需要自己「猜」下一步。

Day 5 新增 `ErrorType` 枚举 + `_ERROR_HINTS` 字典，每种错误都有对应的修复提示：

| 错误类型 | 触发条件 | Hint（模型看到的行动指南） |
|---|---|---|
| `sandbox_escape` | 路径包含 `../` 或绝对路径 | Use list_dir to browse available files |
| `executable_not_allowed` | 调用非白名单命令 | Use web_fetch or web_search instead of curl/wget |
| `dangerous_args` | `python -c` / `-m` / `-i` | Write code to a .py file with write_file, then run it |
| `shell_metacharacter` | 管道、分号等 | Pass one executable and its arguments only |
| `permission_denied` | 用户拒绝了敏感命令 | Try safe read-only commands (ls, cat, grep...) instead |
| `private_address` | URL 指向内网 IP | Try searching for a public equivalent with web_search |
| ... | ... | ... |

### 2.4 各工具改造对比

| 工具 | Day 4 返回 | Day 5 返回 |
|---|---|---|
| `read_file` | 原始字符串（无截断！） | `_make_success({filepath, content, bytes, extension}, truncated, truncated_at)` |
| `write_file` | `{"written": path, "bytes": N}` | `_make_success({filepath, bytes})` |
| `list_dir` | `{"directory": path, "entries": [...]}` | `_make_success({directory, entries})` |
| `run_command` | 抛 `ValueError` 或返回 `{command, exit_code, ...}` | `_make_error(type, msg)` 或 `_make_success({...}, truncated, truncated_at)` |
| `web_fetch` | 抛 `ValueError` 或返回 `{url, title, text, ...}` | `_make_error(type, msg)` 或 `_make_success({...}, truncated, truncated_at)` |
| `web_search` | 抛 `ValueError` 或返回 `{query, results, count}` | `_make_error(type, msg)` 或 `_make_success({query, results, count})` |

### 2.5 Agent Trace 显示升级

Agent 循环也同步升级——解析统一信封，用图标和分类标签展示：

```
# Day 4 (旧): 原始字符串预览
[1] ✅ result: {"query": "Anthropic Building Effective Agents", ...

# Day 5 (新): 解析信封，结构化展示
[1] ✅ 5 results for 'Anthropic Building Effective Agents': ['Building Effective AI Agents...']

# 截断时:
[2] ✅📎 fetched https://www.anthropic.com/... — 10029 chars, status=200
       ⚠️  truncated at 10000 chars

# 错误时:
[3] ❌ [executable_not_allowed] Executable 'env' not in allowlist.
       💡 hint: Allowed executables: cat, echo, ... Use one of these instead.
```

---

## 3. 实验：重新运行 Day 4 多步任务，对比 Trace

### 3.1 实验设置

```bash
python experiments/week03/agent.py --day4
```

- **模型**：`qwen3.7-plus`（通过 DashScope API）
- **工具**：6 个，全部使用 Day 5 统一信封
- **任务**：搜索 Anthropic 文章 → 抓取 → 写摘要 → 三重验证（与 Day 4 完全相同的任务）

### 3.2 Day 5 完整 Trace

```
  ── LLM turn 1 (1 tool call(s)) ──
  [1] 🔧 web_search({"query": "Anthropic Building Effective Agents", "max_results": 5})
  [1] ✅ 5 results for 'Anthropic Building Effective Agents':
         ['Building Effective AI Agents \\ Anthropic', ...]

  ── LLM turn 2 (1 tool call(s)) ──
  [2] 🔧 web_fetch({"url": "https://www.anthropic.com/engineering/building-effective-agents",
                      "max_chars": 10000})
  [2] ✅📎 fetched https://www.anthropic.com/engineering/building-effective-agents —
         title='Building Effective AI Agents \\ Anthropic', 10029 chars, status=200
         ⚠️  truncated at 10000 chars

  ── LLM turn 3 (1 tool call(s)) ──
  [3] 🔧 write_file({"filepath": "notes.md", "content": "# Building Effective AI Agents..."})
  [3] ✅ wrote notes.md (3679 bytes)

  ── LLM turn 4 (1 tool call(s)) ──
  [4] 🔧 read_file({"filepath": "notes.md"})
  [4] ✅ read notes.md (3679 bytes): # Building Effective AI Agents — Summary...

  ── LLM turn 5 (1 tool call(s)) ──
  [5] 🔧 list_dir({"dirpath": "."})
  [5] ✅ .: 7 entries — fetch_url.py, notes.md, notes.txt, ...

  ── LLM turn 6 (1 tool call(s)) ──
  [6] 🔧 run_command({"command": "cat notes.md"})
  [6] ✅ exit=0 — cat notes.md → # Building Effective AI Agents — Summary...
```

Agent 在 **6 个 LLM turn** 内完成了全部 6 个步骤。

### 3.3 Day 4 vs Day 5 Trace 对比

| 维度 | Day 4（旧） | Day 5（新） | 改进 |
|---|---|---|---|
| **结果预览** | JSON 原始字符串（前 250 字符） | 按工具类型生成语义预览 | 模型/人一眼看懂 |
| **截断显示** | 埋在 `data.truncated` 字段中 | 信封级 `✅📎` 图标 + `⚠️ truncated at N chars` | 模型不需要深入 payload 就能发现 |
| **错误格式** | Python 异常字符串 `Error: ...` | 结构化 `[error_type] message` + `💡 hint` | 模型知道**类型**和**下一步怎么修** |
| **read_file 安全** | 无截断保护 | 8,000 chars 截断 | 防止大文件撑爆上下文 |
| **截断一致性** | 3 种截断实现 | 1 个 `_truncate()` | 所有工具截断标记完全相同 |
| **web_fetch** | `max_chars=5000` 硬编码默认值 | 模型传了 `max_chars=10000` 自定义上限 | 模型可以根据任务需要控制抓取长度 |

### 3.4 关键观察

**观察 1：模型主动扩大了抓取上限。** Day 4 中 `web_fetch` 的 prompt 没写 `max_chars`，LLM 也未传，默认用 5,000 chars。Day 5 中模型**主动传入 `max_chars=10000`**——因为 Day 5 的任务 prompt 要求「提取 5 个核心原则」，模型判断 5,000 chars 不够，主动翻倍。结果在 10,000 chars 处仍然截断（`✅📎`），但模型从 `truncated_at` 字段知道发生了截断，可以在最终回答中注明信息不完整。

**观察 2：模型不再只依赖 `max_chars` 默认值。** Day 5 的工具 Schema 明确告知了 `max_chars` 参数的存在和默认值。模型在规划步骤时已经考虑了输出长度。

**观察 3：错误信息不再「裸奔」。** Day 4 中如果工具调用失败，模型看到 `Error: Path escapes sandbox: '../PLAN.md'`。Day 5 中同样的失败会产生：
```json
{
  "status": "error",
  "error": {
    "type": "sandbox_escape",
    "message": "Path escapes sandbox: '../PLAN.md'",
    "hint": "Use list_dir to browse available files. All paths must be relative..."
  }
}
```
模型不仅知道发生了什么（sandbox_escape），还知道**下一步该做什么**（list_dir）。

**观察 4：Trace 的可读性是 Agent 调试的基础。** Day 4 的原始 JSON 预览 vs Day 5 的语义预览——这不仅是美观问题。当 Agent 调用链很深时（搜索 → 抓取 → 写入 → 三重验证），快速理解每一步的结果是调试的关键。

---

## 4. 今日收获

1. **工具输出是 LLM 的「API 协议」。** 就像 REST API 需要一致的 JSON 格式和 HTTP 状态码，工具返回也需要固定的 Schema。不一致的格式相当于每个 API endpoint 返回不同的结构——模型每次都需要重新「解析」。

2. **截断标记是工具输出的「合同条款」。** 统一使用 `… [truncated at N chars]` + 信封级 `truncated_at` 字段，让模型能够：(a) 快速判断信息是否完整，(b) 知道在哪里被截断，(c) 决定是否需要缩小范围重新请求。

3. **错误必须有类型 + 提示，缺一不可。** 错误类型让模型做分类决策（这是权限问题还是参数问题？）；错误提示让模型做行动决策（换工具还是改参数？还是向用户求助？）。Day 5 的 `_ERROR_HINTS` 字典就是这套决策系统的中枢。

4. **截断在 3 个层面的统一：函数、标记、信封。** 所有工具用同一个 `_truncate()` 函数 → 所有截断输出用同一个 `… [truncated at N chars]` 标记 → 所有截断信息在信封级 `truncated` / `truncated_at` 字段声明。三层一致性 = 模型一次学会，永远适用。

5. **最小改动原则：6 个工具，新增 ~150 行，修改 ~200 行。** 没有引入新框架、新依赖、新抽象层。信封函数（`_make_success`、`_make_error`、`_truncate`）不到 50 行，但统一了所有工具的输出契约。

### Day 4 → Day 5 演进

| 维度 | Day 4 | Day 5 |
|---|---|---|
| 工具输出格式 | 6 种不同格式 | 1 种统一信封 |
| 截断实现 | 3 种不同实现 | 1 个 `_truncate()` |
| read_file 截断 | ❌ 无 | ✅ 8,000 chars |
| 错误格式 | Python 异常字符串 | 结构化 `{type, message, hint}` |
| 错误分类 | 无 | 17 种类型 + 对应 hint |
| 模型能否自修复 | 需猜测 | hint 直接告诉它怎么做 |
| Trace 可读性 | JSON 原始字符串 | 语义预览 + 图标 + 分类标签 |
| 截断发现成本 | 需解析 data 字段 | 信封级 `truncated` 字段 |

---

### Done-when 自检

- [x] 实现了统一信封 `_make_success()` / `_make_error()`（5 个顶级字段，所有 6 个工具一致）
- [x] 合并了 3 套截断实现为 1 个 `_truncate()`，标记格式统一为 `… [truncated at N chars]`
- [x] read_file 新增 8,000 chars 截断保护（Day 4 中缺失）
- [x] 17 种 `ErrorType` + `_ERROR_HINTS` 字典，每种错误都有可操作的 hint
- [x] 所有工具函数不再抛出异常，改为返回错误信封（模型侧零异常）
- [x] Agent Trace 升级为语义预览 + 图标（✅/✅📎/❌）+ 错误类型标签 + hint 展示
- [x] System prompt 明确告知模型信封结构 + 截断标记格式 + 错误处理策略
- [x] 运行全部 3 套测试（sandbox / commands / web），所有测试适配新信封格式并通过
- [x] 重新运行 Day 4 多步任务，Agent 正常完成 6 步，trace 质量明显提升
- [x] 对比 Day 4 ↔ Day 5 trace：模型主动传入 `max_chars=10000`，截断标记正确展示

---

## 附录：生产级工具返回设计（扩展阅读）

以下是工具返回设计在生产环境中的完整体系。Day 5 的实验是这套体系的「最小可行版本」——在实际项目中，可以根据需要逐步采纳。
1、Schema 顶层规范（强制统一）
所有工具返回必须遵循统一的顶层信封结构，让模型形成稳定的解析预期
{
  "$schema": "tool_result_v2",
  "envelope": {
    "version": "2.1.0",
    "trace_id": "req_abc123",
    "tool_name": "web_search",
    "invoked_at": "2026-07-31T15:04:00Z",
    "latency_ms": 245,
    "status": "success"
  },
  "payload": { ... },
  "meta": { ... },
  "error": null,
  "guidance": { ... }
}
2、Payload 设计原则
业务数据与元数据严格分离
{
  "payload": {
    "answer": "巴黎是法国首都",
    "confidence": 0.98,
    "sources": [
      {"title": "France - Wikipedia", "url": "https://...", "reliability": "high"}
    ]
  },
  "meta": {
    "result_count": 1,
    "search_engine": "bing",
    "region": "fr-FR"
  }
}
禁止把 search_engine 或 latency 塞进 payload，那会污染模型的业务理解。
空值显式化策略
LLM 依赖稳定的字段存在性做推理。字段忽有忽无会导致模型幻觉填充。
| 场景      | 正确做法                                                       | 错误做法                      |
| ------- | ---------------------------------------------------------- | ------------------------- |
| 查询无结果   | `{"payload": {"results": [], "empty_reason": "NO_MATCH"}}` | `{"payload": null}` 或完全省略 |
| 字段无权限   | `{"field_x": null, "field_x_status": "FORBIDDEN"}`         | 直接不出现该字段                  |
| 可选字段未填充 | `{"optional_note": null}`                                  | 省略字段导致 Schema 不稳定         |
类型严格性
* 禁止把所有东西包成字符串。"count": "42" 是灾难。
* 枚举值必须是大写下划线格式："status": "RATE_LIMITED"，而不是 "status": "rate limit reached"。
* 时间统一 ISO 8601 + UTC："2026-07-31T07:04:00Z"。
* 金额/数值必须带单位或明确语义：{"value": 100.5, "currency": "USD"}。
3、Meta 层：可观测性与上下文控制
{
  "meta": {
    "pagination": {
      "has_more": true,
      "current_page": 1,
      "page_size": 10,
      "total_estimated": 5000,
      "next_cursor": "eyJpZCI6MTB9"
    },
    "truncation": {
      "is_truncated": true,
      "original_size": 150000,
      "returned_size": 4000,
      "truncation_strategy": "HEAD_TAIL",
      "omitted_sections": ["references", "appendix"]
    },
    "provenance": {
      "sources": [
        {
          "id": "src_001",
          "type": "web_page",
          "url": "https://...",
          "fetched_at": "2026-07-31T07:00:00Z",
          "cache_ttl_seconds": 3600,
          "source_reliability": "high"
        }
      ],
      "data_freshness": "realtime"
    },
    "cost": {
      "tokens_consumed": 1200,
      "api_calls_made": 3,
      "estimated_usd": 0.0042
    }
  }
}
* truncation 必须显式声明，否则模型会以为看到了完整数据。
* provenance 让模型知道信息来源的时效性和可信度，避免用过期数据推理。
* cost 帮助模型在资源受限时自我节制（例如「已消耗 3000 tokens，建议精简后续查询」）
4、Guidance 层：LLM 行动协议（核心创新）
{
  "guidance": {
    "summary": "搜索到 3 条相关结果，其中 2 条来自官方文档，1 条来自社区博客。",
    "caveats": [
      "结果 #3 发布于 2023 年，可能已过时",
      "官方文档链接需要登录才能查看完整内容"
    ],
    "suggested_next_steps": [
      {
        "action": "FETCH_DETAIL",
        "target": "src_001",
        "reason": "官方文档最权威，建议提取完整内容",
        "expected_cost": "约 800 tokens"
      },
      {
        "action": "REFINE_QUERY",
        "query": "Anthropic Building Effective Agents 2026 update",
        "reason": "当前结果可能缺少最新更新"
      }
    ],
    "confidence": {
      "overall": 0.85,
      "factors": [
        {"dimension": "source_authority", "score": 0.95},
        {"dimension": "recency", "score": 0.70}
      ]
    }
  }
}
设计原则：
* summary：用 1-2 句话让模型快速建立上下文，避免它自己从 payload 里「总结」。
* caveats：显式标注数据的陷阱（过期、权限、偏见），防止模型盲目采信。
* suggested_next_steps：结构化行动建议，包含 action（动作类型）、reason（理由）、expected_cost（成本预估）。模型可以据此做理性决策。
* confidence：让模型知道何时应该「质疑」结果，而不是无脑引用。
完备的错误设计体系
[调用开始]
    │
    ▼
[参数校验] ──参数非法──► ERROR (permanent/validation)
    │
    ▼
[权限校验] ──无权限───► ERROR (auth/forbidden)
    │
    ▼
[执行中] ────超时────► ERROR (transient/timeout)
    │         │
    │         └───可重试？───► 自动重试 ──► 仍失败 ──► ERROR (transient/max_retries)
    │
    ▼
[部分成功] ─────────► PARTIAL (部分数据可用，部分失败)
    │
    ▼
[完全成功] ─────────► SUCCESS

错误分类体系（三层层次化）
| 层级           | 字段               | 职责             | 示例                                                                    |
| ------------ | ---------------- | -------------- | --------------------------------------------------------------------- |
| **Domain**   | `error.domain`   | 错误发生的系统边界      | `TOOL_EXECUTION` / `UPSTREAM_API` / `NETWORK` / `AUTH` / `VALIDATION` |
| **Category** | `error.category` | 错误的业务语义类别      | `RATE_LIMITED` / `NOT_FOUND` / `TIMEOUT` / `INVALID_PARAM`            |
| **Type**     | `error.type`     | 具体错误子类型，用于精确处理 | `RATE_LIMITED__SECONDARY` / `NOT_FOUND__RESOURCE_MOVED`               |
* Domain 告诉模型「这是谁的问题」；
* Category 告诉模型「这是什么性质的问题」；
* Type 告诉系统（不是模型）「该走哪条修复分支」。

修复指导设计原则（Remediation）
| 维度       | 字段             | 说明                         |
| -------- | -------------- | -------------------------- |
| **自动修复** | `auto_retry`   | 是否适合自动重试？如果适合，提供退避策略       |
| **降级路径** | `fallbacks`    | 不依赖用户介入的替代方案（换工具、用缓存、缩小范围） |
| **用户介入** | `user_clarify` | 需要用户决策时的提示文案和选项            |
| **权宜之计** | `workarounds`  | 在不解决根本问题的情况下，如何尽量推进任务      |
| Severity    | 含义            | 系统行为        | 模型行为          |
| ----------- | ------------- | ----------- | ------------- |
| `transient` | 临时故障，可能自恢复    | 自动重试（指数退避）  | 等待，必要时汇报延迟    |
| `degraded`  | 主功能受损，但可用降级方案 | 触发 fallback | 主动选择替代路径继续    |
| `permanent` | 逻辑错误，重试无用     | 直接报错        | 向用户解释，请求修正输入  |
| `auth`      | 权限/认证问题       | 不自动重试       | 引导用户授权或降低权限要求 |


示例：
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ToolResult",
  "type": "object",
  "required": ["envelope", "payload", "meta", "error", "guidance"],
  "properties": {
    "envelope": {
      "type": "object",
      "required": ["version", "trace_id", "tool_name", "invoked_at", "status"],
      "properties": {
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "trace_id": {"type": "string", "minLength": 1},
        "tool_name": {"type": "string"},
        "invoked_at": {"type": "string", "format": "date-time"},
        "latency_ms": {"type": "integer", "minimum": 0},
        "status": {"enum": ["success", "partial", "error", "degraded"]}
      }
    },
    "payload": {"type": ["object", "array", "null"]},
    "meta": {
      "type": "object",
      "required": ["truncation", "provenance"],
      "properties": {
        "pagination": {
          "type": "object",
          "properties": {
            "has_more": {"type": "boolean"},
            "next_cursor": {"type": ["string", "null"]}
          }
        },
        "truncation": {
          "type": "object",
          "properties": {
            "is_truncated": {"type": "boolean"},
            "original_size": {"type": "integer"},
            "returned_size": {"type": "integer"}
          }
        },
        "provenance": {"type": "object"},
        "cost": {
          "type": "object",
          "properties": {
            "tokens_consumed": {"type": "integer"},
            "api_calls_made": {"type": "integer"}
          }
        }
      }
    },
    "error": {
      "oneOf": [
        {"type": "null"},
        {
          "type": "object",
          "required": ["domain", "category", "severity", "message", "remediation"],
          "properties": {
            "domain": {"enum": ["TOOL_EXECUTION", "UPSTREAM_API", "NETWORK", "AUTH", "VALIDATION"]},
            "category": {"type": "string"},
            "type": {"type": "string"},
            "severity": {"enum": ["transient", "degraded", "permanent", "auth"]},
            "code": {"type": "string"},
            "message": {
              "type": "object",
              "properties": {
                "human": {"type": "string"},
                "llm": {"type": "string"}
              }
            },
            "context": {"type": "object"},
            "remediation": {
              "type": "object",
              "properties": {
                "auto_retry": {
                  "type": "object",
                  "properties": {
                    "eligible": {"type": "boolean"},
                    "backoff_seconds": {"type": "integer"},
                    "max_attempts": {"type": "integer"}
                  }
                },
                "fallbacks": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "strategy": {"enum": ["SWITCH_TOOL", "USE_CACHE", "REDUCE_SCOPE", "USER_CLARIFY"]},
                      "description": {"type": "string"},
                      "trade_off": {"type": "string"}
                    }
                  }
                }
              }
            },
            "impact": {
              "type": "object",
              "properties": {
                "scope": {"enum": ["total", "partial"]},
                "affected_payload_paths": {"type": "array", "items": {"type": "string"}}
              }
            }
          }
        }
      ]
    },
    "guidance": {
      "type": "object",
      "properties": {
        "summary": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "suggested_next_steps": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "action": {"type": "string"},
              "reason": {"type": "string"},
              "expected_cost": {"type": "string"}
            }
          }
        },
        "confidence": {
          "type": "object",
          "properties": {
            "overall": {"type": "number", "minimum": 0, "maximum": 1},
            "factors": {"type": "array"}
          }
        }
      }
    }
  }
}
但是过于完备的工具设计可能很容易引起上下文爆炸的问题，所以需要一些机制解决：
如工具本身支持返回粒度控制，由调用方在调用时指定。
| 级别         | 适用场景      | 返回大小             | 内容                                                     |
| ---------- | --------- | ---------------- | ------------------------------------------------------ |
| `compact`  | 快速探索、多轮调用 | ~200-500 tokens  | 只有 `envelope` + `payload` 摘要 + `guidance.summary`（一句话） |
| `standard` | 常规任务      | ~800-1500 tokens | 完整 `payload` + 精简 `meta` + 完整 `guidance`               |
| `verbose`  | 深度分析、调试   | ~2000+ tokens    | 完整Schema，含 `provenance`、`cost`、`error.context` 等       |
历史工具结果的「摘要压缩」（History Compression）
# 伪代码：对话历史中的工具结果压缩
for tool_call in history:
    if tool_call.age > 3_turns:  # 3 轮之前的工具调用
        # 将完整结果替换为「引用摘要」
        tool_call.result = {
            "type": "reference",
            "trace_id": tool_call.trace_id,
            "summary": tool_call.result.guidance.summary,  # 只保留一句话摘要
            "key_finding": extract_key_fact(tool_call.result.payload),
            "full_result_available": true  # 告诉模型完整数据还在，需要时可以展开
        }
按需展开」机制（On-Demand Expansion
模型不需要一直背着所有历史数据。它只需要一个「目录」：
{
  "available_contexts": [
    {
      "trace_id": "req_def456",
      "tool": "web_fetch",
      "summary": "官方博客原文，含 5 条核心原则",
      "size_tokens": 3500,
      "access_command": "expand_context('req_def456')"
    },
    {
      "trace_id": "req_abc123", 
      "tool": "web_search",
      "summary": "搜索结果列表",
      "size_tokens": 800,
      "access_command": "expand_context('req_abc123')"
    }
  ],
  "current_context_budget": "已用 12k / 32k tokens，剩余 20k"
}
当模型需要某个历史细节时，它主动请求展开：
"我需要查看 req_def456 的完整内容来回答第 3 条原则。"
系统将该次调用的完整 payload 注入当前上下文，同时可能压缩其他不相关的历史。
Guidance 是工具开发者对模型的「耳语」，不是模型的自言自语；返回长度通过「分层返回 + 历史压缩 + 按需展开」三件套控制——完备性留给 Schema 定义，实际传输时只给模型当下最需要的那一层信息。

---

# Week 3 学习报告 — Day 6

## 今日实验：工具数量对任务成功率的影响 — 相同任务，4 个工具 vs. 4+8 个无关工具

---

## 1. 实验问题

**一个很自然的直觉是**：给 Agent 的工具越多，它选错工具的概率越大。但这条直觉需要量化——到底会不会错、错多少次、代价是什么？

**实验问题（只改一个变量）**：相同的任务，用 4 个核心工具（`read_file` / `write_file` / `list_dir` / `run_command`）跑 vs. 用这 4 个 + 8 个无关工具（`web_search` / `web_fetch` / `send_email` / `create_calendar_event` / `translate_text` / `generate_image` / `query_database` / `calculate_hash`）跑。任务的正确答案完全依赖那 4 个核心工具——8 个额外工具是纯粹的噪声。

**实验前先立假设**（避免"看到结果再编解释"）：多出 8 个看似相关的工具（比如 `query_database` 会诱惑模型不去读 JSON 文件、`send_email` 会诱惑模型发送结果而非写文件），12-tools 条件下应该出现一定比例的**错误工具调用**，进而拉低成功率或至少拉高 token 消耗（失败重试）。

实验代码：`experiments/week03/day6_tool_count.py`。自动判分器：`expected_substring`（所有期望子串都在答案里才算通过）。每个条件 × 5 个任务 × 5 次运行 = **50 次 agent 运行**。模型：`qwen3.7-plus`。

**5 个 Eval 任务（全部只依赖 4 个核心工具即可完成）：**

| 任务 ID | 说明 | 需要的核心工具 | 期望子串 |
| --- | --- | --- | --- |
| `count-lines` | 用 `run_command` 统计 `notes.txt` 行数 | `run_command` | `4` |
| `read-json` | 读 `subdir/data.json`，返回 project 字段值 | `read_file` | `ForLearning` |
| `list-dir` | 列 sandbox 根目录，找 `.md` 文件 | `list_dir` | `notes.md` + `python_website_notes.md` |
| `write-then-verify` | 写文件 → 读回验证 | `write_file` + `read_file` | `experiment complete` |
| `grep-md-files` | 递归查找所有 `.md` 文件 | `run_command` | `notes.md` + `python_website_notes.md` |

8 个无关工具的定义策略：描述听起来像真工具（模型应感知它们为合法选项而非明显诱饵），但返回结果要么是模拟的、要么明确提示"数据不在这里，请用 sandbox 文件工具"。

---

## 2. 实验结果

### 2.1 总体通过率

| 条件 | 工具数 | 通过 / 总数 | 成功率 |
| --- | --- | --- | --- |
| A（4-tools） | 4 | 25 / 25 | **100%** |
| B（12-tools） | 12 | 25 / 25 | **100%** |

**两条线完全重合——假设被推翻了。** 至少在当前模型（qwen3.7-plus）和当前任务难度下，8 个无关工具没有导致哪怕一次错误工具调用。模型完美地避开了所有无关工具，每次只选择正确的核心工具。

### 2.2 每个任务的通过次数（各 5 次）

```text
task                  4-tools     12-tools
------------------------------------------
count-lines               5           5
read-json                 5           5
list-dir                  5           5
write-then-verify         5           5
grep-md-files             5           5
```

全部 5/5，零波动——跟 Week 2 Day 6 的温度实验形成鲜明对比（那里至少有几处 4/5）。

### 2.3 Token 消耗（这才是真正的代价）

| 条件 | 平均 tokens/次 | 总 tokens（25 次） | 增幅 |
| --- | --- | --- | --- |
| 4-tools | **2,770** | 69,246 | — |
| 12-tools | **5,635** | 140,866 | **+103%（约 2.03×）** |

**Schema 税**：即使模型一次都没用错工具，12-tools 的每次 API 调用都在 system prompt 和 tool schemas 里多传了 8 个工具定义（每个约 200-400 字符的 JSON Schema）——这些字节每轮循环都要重新发送一遍。成功的代价不在准确率，全在 token 账单上。

### 2.4 按任务拆解 token 消耗

| 任务 | 4-tools 均值 | 12-tools 均值 | 倍数 |
| --- | --- | --- | --- |
| `count-lines` | 1,790 | 3,739 | 2.09× |
| `read-json` | 1,809 | 3,756 | 2.08× |
| `list-dir` | 2,000 | 4,027 | 2.01× |
| `write-then-verify` | 2,941 | 5,885 | 2.00× |
| `grep-md-files` | **5,310** | **10,765** | **2.03×** |

单步任务（1 个 tool call）的倍数基本稳定在 ~2.0-2.1×，因为 schema 开销是固定的（每次 API 调用的 tool 定义区域）。`grep-md-files` 因为是**多步任务**（3-5 个 tool call = 4-6 个 API 调用），schema 税被循环放大了。

### 2.5 无关工具调用统计（Condition B）

| 指标 | 值 |
| --- | --- |
| 无关工具总调用次数 | **0** |
| 含 ≥1 次无关调用的 run 数 | 0 / 25 |
| 最常被误调的无关工具 | n/a（没有一次被调用） |

模型在 12-tools 下从未调用过 `web_search`、`web_fetch`、`send_email`、`query_database`、`calculate_hash`、`translate_text`、`generate_image` 或 `create_calendar_event`。它坚定地只用 `read_file`、`write_file`、`list_dir`、`run_command` 四个核心工具。

---

## 3. 分析

### 发现 1：假设被干净地推翻了——模型比预想的更"专注"

这个结果跟直觉完全相反。在实验前我预期至少会有几次 `query_database`（"读 data.json 不如直接查数据库"）或 `send_email`（"写完了要不要发出去"）的调用——尤其是在 `read-json` 和 `write-then-verify` 这两个任务上，无关工具的"语义邻近性"（database ↔ data file；email ↔ 分享文件结果）理论上最容易触发混淆。

**为什么没发生？** 结合 trace 分析，几个可能的解释：

1. **任务 prompt 的指向性足够强。** 每个任务的 prompt 都明确指定了"用 run_command"、"读 subdir/data.json"等——不给模型太多需要"自己猜用什么工具"的模糊空间。这吻合"Building Effective Agents"的核心原则：清晰的指令降低工具选择的不确定性。

2. **工具描述的 when-to-call 条件在起作用。** Day 5 重构后的工具描述都加了明确的触发条件（"Call when the user asks to see or summarize a file"），这让模型能在 prompt 中快速匹配"读文件 → read_file"、"列目录 → list_dir"的映射，而不被无关工具的泛化描述误导。

3. **qwen3.7-plus 在工具选择上的表现可能比预期好。** 更小或更老的模型在这个实验上可能表现不同——这是下一步可以探索的方向。

4. **任务太简单了。** 5 个任务都是 1-3 步的直线任务，不存在"我该不该多做一个步骤"的模糊空间。如果把任务改成开放式（"帮我看看 sandbox 里有什么有用的东西"），无关工具的诱惑力可能会变大。

### 发现 2：真正的成本不是错误，是 Schema 税

**这是本实验最重要的发现。** 同等成功率下，12-tools 的 token 消耗是 4-tools 的 **2.03 倍**——而这 2 倍不是浪费在"试错重试"上，而是浪费在"每次 API 调用都必须传输 12 个 tool schema"上。

```text
一个 tool schema 约 200-400 字符（~100-200 tokens）
8 个额外 schema ≈ 800-1600 tokens
每次 API 调用都重复传输
多步任务（grep-md-files）: 4-6 次调用 → schema 税 × (4-6)
```

这就是 Week 5 Day 3 要量化的：**工具 schema 本身的上下文成本**。本实验给了一个预览：在 4→12 这个区间，token 账单翻倍，而成功率纹丝不动。

**一个推论**：如果一个生产 agent 每天处理 10 万次请求，12-tools 方案会比 4-tools 方案多烧 ~100% 的 prompt token 费用——而这笔钱**没有换来任何准确率收益**。工具数量的 ROI 曲线在某个点上急剧下降，本实验暗示这个拐点可能比预想的更靠左。

### 发现 3：grep-md-files 的隐蔽差异——同样的成功率，不同的"策略质量"

虽然两个条件都是 5/5 通过，但 12-tools 条件下模型的工具使用策略出现了微妙退化：

```text
find 命令调用次数（grep-md-files 任务）:
  4-tools:  0/5 次
  12-tools: 3/5 次
```

`find` 不在 allowlist 里（只有 `ls, cat, echo, grep, wc, head, tail, python, python3`），调用会直接报错。在 4-tools 条件下模型直接使用 `python3 -c "..."` 完成递归搜索；在 12-tools 条件下 3/5 次先尝试了 `find`，遇到 `executable_not_allowed` 错误后才切换回 `python3`。

**为什么会这样？** 这不是模型"选错了工具"——两次都是调用 `run_command`，工具没有选错。差异在 `run_command` 的**参数层面**：12-tools 条件下，run_command 的 description 在 12 个 schema 的大海里只是一句话（"Only allowlisted executables are permitted…"），模型可能没有充分消化这条约束。换句话说，**工具数量多了，每个工具 description 中的关键约束被稀释了**。

这解释了为什么没有"无关工具调用"但有"次优参数选择"：工具选择的准确率（"我该调哪个工具"）没受影响，但**工具使用的精确度**（"该怎么调这个工具"）受到了 schema 噪声的轻微干扰。

> 这是一个重要的区分：**tool selection** vs **tool usage**——选对工具 ≠ 用对参数。增加工具数量的副作用可能更早地体现在 tool usage 层面，而不是 tool selection 层面。

### 发现 4：模型从 allowlist 错误中恢复的能力很强

3 次 `find` 错误后，模型在下一轮都正确切换到了 `python3 -c "..."` 或 `python3 find_md.py`（写脚本再执行）。没有一次因为 `find` 失败而卡住或放弃——这验证了 Day 4/Day 5 错误提示设计的有效性：`executable_not_allowed` 的 error.hint 明确告诉模型"Allowed executables: ls, cat, echo, grep, wc, head, tail, python, python3"——模型读了 hint 就能立刻纠正。

**但恢复有代价**：每次失败的 `find` 调用多消耗一轮 API 调用（~2000-3000 tokens），且错误信息和重试指令注入上下文，让后续每轮都变贵。这是 12-tools 下 `grep-md-files` token 消耗方差更大的直接原因：

```text
grep-md-files token 分布:
  4-tools:  4,824 - 6,883  (范围 ~2,060)
  12-tools: 8,697 - 13,840 (范围 ~5,140)
```

> **错误恢复能力强 ≠ 免费。** 每次恢复都在烧 token——在当前实验里这些小错误不影响成功率，但在更严苛的成本预算或更长的任务链路上，积累的"恢复税"可能会吃掉整个准确率优势。

---

## 4. 收获

1. **无关工具不一定会导致错误工具调用——至少在这些任务和这个模型上。** 模型在工具选择上比直觉预测的更稳健。但稳健不等于零成本——Schema 税在工具数量翻 3 倍时让 token 账单也翻了倍。把工具数量从 4 减到"刚好够用"是 ROI 最高的优化。

2. **区分 tool selection 和 tool usage 的退化模式。** 本实验中 tool selection 完美（从未选错工具），但 tool usage（参数选择）在 12-tools 下出现了可测量的退化（`find` 调用增多）。这意味着把关注点从"有没有选错工具"扩展到"有没有用对参数"——后者可能是工具数量增加时更先出问题的维度。

3. **Schema 税不是小数字。** 8 个额外 tool schema 让每次 API 调用贵了 ~100%。这跟模型有没有选错工具无关——Schema 税是固定开销，多步任务里被循环放大。Week 5 Day 3 要专门量这个（5 / 15 / 30 个工具下的 token 消耗曲线）。

4. **错误恢复能力在"兜底"，但不应依赖它。** Day 5 的可操作错误提示确实有效——模型每次都能从 `find` 错误中恢复。但恢复本身有 token 代价，且在更复杂任务中一次恢复失败就可能引发连锁反应。好的工具描述应该让模型"第一次就做对"，而不是依赖"错了能改"。

5. **n=5、5 个简单任务的局限性——下一步需要更难的实验。** 本次实验的"天花板效应"（满分通过）限制了发现的深度——看不出两个条件在更复杂场景下的真实差距。下一步应该用**更模糊的 prompt**（不给明确的工具名称提示）和**更多步的开放式任务**来放大工具数量带来的选择压力。

---

## 5. 思考

1. **如果把 prompt 里的工具名提示去掉会怎样？** 本实验的每个任务 prompt 都写了"Use run_command to…"或"Read the file…"，等于替模型做了工具选择。如果把 prompt 改成"沙箱里有一个 notes.txt 文件。告诉我它有几行。"——不给任何工具提示——无关工具的干扰效应可能才会真正显现。**下一步实验应该对比"有/无工具提示的 prompt"×"4/12 个工具"。**

2. **Schema 税可以用"按需加载工具"来规避吗？** 如果系统先让模型用一个 `select_tool_category` 做轻量路由（类似 Week 5 Day 5 的 router 模式），只把 4 个相关工具的 schema 注入实际调用，token 成本可以回到接近 4-tools 的水平。代价是多一次路由调用——值得在 Week 5 实验里对比"router 模式 vs. 全量工具"的 token × 准确率 trade-off。

3. **更小的模型会怎样？** qwen3.7-plus 在工具选择上表现很好，但 qwen-turbo 或更小的开源模型可能更容易被多余工具干扰。同一个实验在不同模型上跑一遍，可能会看到从"零干扰"到"可测量下降"的转折点——这对生产选型很重要（省钱的小模型能不能扛住 30 个工具的场景？）。

4. **工具的"语义邻近性"可能比数量更重要。** 本实验的 8 个无关工具跟任务需求有明确的语义距离（数据库查询 vs. 读本地文件）。如果换成 8 个"看起来很像"的工具——比如 `read_json_file`、`read_csv_file`、`read_text_file` 三个变体而非一个 `read_file`——即使总数不变，干扰效应可能大得多。**工具质量的维度不只是"多少"，还有"多像"。**

5. **这次实验最诚实的一句话**：在 qwen3.7-plus + 明确 prompt + 简单直线任务 的组合下，12 个工具没有降低成功率。但这个结论的外部效度（能推广到其他场景吗？）需要更多实验来验证——更模糊的 prompt、更复杂的任务、更小的模型、语义更邻近的干扰工具，每个变量都可能改变答案。Day 6 的价值不在"证明了什么"，而在**排除了一个常见假设（"工具多=乱选"至少在这个条件下不成立）并指明了下一条要追的线索（Schema 税 + tool usage 退化 + 语义邻近性）**。

---

## Day 7 — Eval 基线：把 Week 3 的能力固化进 eval 套件

### 今日主题

按照 PLAN.md 的 Day 7 节奏——**写报告 + 加 eval 任务 + 跑基线**。Day 7 不是学新东西，是把这一周的能力装进 `evals/tasks.json`，让未来每一周都能自动检测"Week 3 级别的能力有没有退化"。

---

### 1. 新增的 Eval 任务

向 `evals/tasks.json` 新增了 6 个任务，覆盖 Week 3 的核心工具能力：

| 任务 ID | 测试能力 | 需要的工具 | 验证方式 |
| --- | --- | --- | --- |
| `sandbox-list-py` | 沙箱目录浏览 | `list_dir` | 回答 ".py 文件数量 = 3" |
| `sandbox-read-json` | 沙箱文件读取 | `read_file` | 从 JSON 中提取 `project` 字段 |
| `sandbox-write-verify` | 文件写入 + 验证 | `write_file`, `read_file` | 写 → 读 → 确认内容一致 |
| `sandbox-count-lines` | 命令执行 | `run_command` | `wc -l` 统计行数 |
| `sandbox-grep-urllib` | 命令执行（搜索） | `run_command` | `grep` 搜索文件内容 |
| `sandbox-multi-step` | 多步工具链 | `list_dir`, `read_file` | 先浏览目录 → 再读文件 → 提取首行 |

加上原有的 `no-tool-needed`（跨 agent 兼容性基准），Week 3 eval 套件共 **7 个任务**。

同时更新了 `evals/run_evals.py`，新增 `--agent` 参数支持在 week01 和 week03 agent 之间切换，week01 agent 自动跳过 sandbox 系列任务。

---

### 2. 实验结果

```text
Model:    qwen3.7-plus
Tasks:    7
Runs/task: 5
Sandbox:  experiments/week03/sandbox/

Total: 35/35 passed (100%)
Avg tokens/run: 3,389
Total est. cost: $0.0512
```

| 任务 | 通过 | 平均 tokens | 工具调用模式 |
| --- | --- | --- | --- |
| `no-tool-needed` | 5/5 ✅ | 1,266 | 零工具调用（1 turn） |
| `sandbox-list-py` | 5/5 ✅ | 2,892 | `list_dir` ×1（2 turns） |
| `sandbox-read-json` | 5/5 ✅ | 2,684 | `read_file` ×1（2 turns） |
| `sandbox-write-verify` | 5/5 ✅ | 4,272 | `write_file` → `read_file`（3 turns） |
| `sandbox-count-lines` | 5/5 ✅ | 2,660 | `run_command` ×1（2 turns） |
| `sandbox-grep-urllib` | 5/5 ✅ | 5,885 | `run_command` ×1–5（2–4 turns） |
| `sandbox-multi-step` | 5/5 ✅ | 4,066 | `list_dir` → `read_file`（2–3 turns） |

**35/35 = 100% 通过率**，在所有 7 个任务上 5 次运行全部正确。

---

### 3. 分析

#### 3.1 单步工具任务稳定且高效

`sandbox-list-py`、`sandbox-read-json`、`sandbox-count-lines` 这三个任务都是 1 次工具调用、2 turns 完成——模型看到 prompt → 选对工具 → 拿到结果 → 直接回答。每 run 约 2,600–2,900 tokens，方差极小。说明当任务明确提示"用 X 工具做 Y 事"时，沙箱 agent 的行为高度可预测。

#### 3.2 sandbox-grep-urllib：正确但昂贵的"探索式"行为

这是 7 个任务中**最贵的一个**（平均 5,885 tokens/run，是最便宜任务的 2.2 倍），且方差最大：

```text
sandbox-grep-urllib token 分布:
  高效路径（1/5）: 1 × run_command → 2,696 tokens
  探索路径（4/5）: list_dir + 多次 run_command 试错 → 6,597–6,904 tokens
```

在 4/5 的运行中，模型采取了迂回策略：先用 `list_dir` 浏览所有文件，然后多次尝试不同的 grep/python 命令（`grep -r "urllib" .`、`python3 -c "..."`、`grep "urllib" *.py` 等），最终才收敛到正确答案。虽然每次都对了，但多烧了 ~4,000 tokens。

这跟 Day 6 的发现一致——**tool usage（参数选择）比 tool selection（选哪个工具）更容易出问题**。模型知道要用 `run_command`（tool selection 正确），但不确定该传什么 command 参数（tool usage 有噪音），于是用"多试几次"来对冲不确定性。

对比 `sandbox-count-lines`（prompt 里暗示了 `wc -l`），`sandbox-grep-urllib` 的 prompt 只说"use run_command with grep to find..."——语气更开放，没有给出具体的命令模板。这给了模型更多"探索空间"，也带来了更多 token 浪费。**一个可能的结论：prompt 中的工具使用提示越具体，模型越省 token——但这也会降低任务的"测试纯度"（你在测试工具使用能力，还是在测试 prompt 给出的指令遵循能力？）。**

#### 3.3 多步工具链：模型自动串联不同工具

`sandbox-multi-step` 要求 `list_dir` → `read_file` → 提取首行。所有 5 次运行都正确完成了这个链条——没有一次跳过 `list_dir` 直接 `read_file`，也没有一次在 `list_dir` 之后就回答。模型严格按 prompt 要求的两步顺序执行。

在 2/5 的运行中模型在 turn 1 同时调用了两个工具（并行 tool calling），3/5 的运行中串行调用。并行调用时 token 更省（~3,080 vs ~4,720），因为少了一轮 API 往返。这说明当两个工具调用互不依赖时，模型有时能识别并利用并行能力——不过当前的并行率（2/5）还比较随机。

#### 3.4 与 Day 6 的对比

| 维度 | Day 6 实验 | Day 7 eval |
| --- | --- | --- |
| 工具集 | 4 或 12 个工具 | 6 个工具（全部有用） |
| 任务数 | 5 | 7 |
| 单任务 runs | 5 | 5 |
| 通过率 | 100%（两个条件） | 100% |
| 平均 tokens/run | 2,770（4-tools）/ 5,635（12-tools） | 3,389（6-tools） |

Day 7 的 3,389 tokens/run 介于 Day 6 的 4-tools 和 12-tools 之间——比 4-tools 多 ~22%，比 12-tools 少 ~40%。这符合直觉：6 个全有用的工具在 schema 开销上多于 4 个精挑细选的，但远少于 12 个。

---

### 4. 收获

1. **Eval 套件是能力的"回归测试"。** 6 个新任务覆盖了 Week 3 的每个核心工具（file / command / multi-step），未来任何 agent 改动都能通过 `python evals/run_evals.py --agent week03` 一行命令验证"Week 3 级别的基本功"还在不在。

2. **100% 通过率说明任务偏简单，但不说明没用。** 这些任务的价值不在"挑战模型"，而在"当某天通过率从 100% 掉到 80% 时，你知道什么东西坏了"。好的回归测试就该简单、稳定、快速——过不了就是真的出问题了。

3. **`sandbox-grep-urllib` 的 5,885 tokens 是值得关注的"效率基线"。** 如果 Week 5 做了工具描述优化后这个数字明显下降，那就证明描述质量的改善有量化的 token 成本收益。留着这条基线。

4. **Prompt 的"工具提示密度"影响 token 效率。** 给模型明确的工具使用提示（"Use run_command with wc -l"）vs. 模糊提示（"Use run_command with grep to find..."），后者让模型多烧了 ~4,000 tokens 来"探索"。在设计 eval 任务时，**提示的具体程度本身就是一个实验变量**——需要在"测试工具能力"和"降低探索噪音"之间权衡。

---

## Week 3 总结

### 这一周做了什么

| Day | 主题 | 产出 |
| --- | --- | --- |
| 1 | 路径遍历攻击与沙箱文件工具 | `read_file` / `write_file` / `list_dir` + `resolve_sandbox_path()` |
| 2 | 命令注入与 allowlist 防御 | `run_command`（7 层安全防护）+ 可执行文件白名单 |
| 3 | Web 工具 | `web_fetch` + `web_search` + 私网 IP 检测 |
| 4 | 组合所有工具，多步任务 | Day 4 trace：search → fetch → write → read → verify |
| 5 | 工具结果格式统一化 | 统一 JSON envelope + ErrorType 分类 + error.hint |
| 6 | **实验：工具数量 vs. 成功率** | 4 tools vs. 12 tools → 成功率相同，token 翻倍 |
| 7 | **Eval 基线 + 报告** | 新增 6 个沙箱任务到 `evals/tasks.json`，35/35 全通过 |

### Week 3 核心认知

从一个"只能算数和看时间的玩具 agent"变成了一个**有文件系统、shell 执行、网络访问能力的真实 agent**——同时也理解了这些能力伴随的安全边界（沙箱、allowlist、私网拦截）和设计决策（工具粒度、结果格式、错误可操作性）。

**最大的收获**：工具是 agent 的语言。模型通过工具跟世界交互——工具的 schema 定义了什么可以表达（name/description/parameters），工具的输出定义了模型能看到什么（格式/截断/错误信息）。做好工具设计就是做好 agent 的"感官系统"：给多了浪费（Schema 税），给少了致盲（任务做不了），给错了误导（选错工具 / 用错参数）。Day 6 的实验给这个 trade-off 画了一条具体的线——至少在简单任务上，4 个刚好够用的工具 = 12 个工具的准确率，但便宜一半。

Week 3 的代码和 eval 基线将成为 Phase 2（Week 4–8）的基础——从 Week 4 开始的所有能力（eval 体系、工具设计、上下文管理、记忆、MCP）都将在 Week 3 的 agent 之上构建和度量。

