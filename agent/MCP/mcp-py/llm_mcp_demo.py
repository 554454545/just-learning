"""LLM 通过 MCP 调用外部工具（懒加载版，对齐 Hermes mcp_tool.py）。

架构：
  ① McpProtocol    —— 协议层（tools/resources/prompts/ping，分页，多模态，安全扫描）
  ② StdioMcpClient —— stdio 传输（本地子进程 + recv loop 分发）
  ③ HttpMcpClient  —— 远程传输（streamable HTTP，aiohttp POST）
  ④ MCPManager     —— 懒加载管理（对齐 Hermes 的 lazy start on first use）

懒加载机制（对齐 Hermes）：
  - 启动时：有 schema 缓存（磁盘 .mcp_schema_cache.json）的 server
    只注册工具名、不启动进程（对齐 lazy_registered，6305 行）
    无缓存的并发连接（首次总要拉一次工具清单建缓存）
  - 调用时：_ensure_server 现场检查（对齐 lazy_start_server，4781 行）
    活着 → 直接用；死了 → 现场重连；冷却中/连接中 → 本次调用失败
  - connect cooldown：失败后冷却期不反复重试（对齐 _connect_cooldown_active）
  - _connecting 去重集：同一 server 只允许一个连接任务（对齐 _server_connecting）

运行：
  python3 llm_mcp_demo.py "看看 zsrc 目录下有哪些 .py 文件"
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────
ZSRC = Path("/home/judy/zh/agent/hermes-agent-main/zsrc")
ENV = ZSRC / ".env"
MCP_CFG = Path(__file__).parent / "mcp.json"
SCHEMA_CACHE_FILE = Path(__file__).parent / ".mcp_schema_cache.json"

# 读 DeepSeek key（zsrc/.env）
_api_key = ""
for line in ENV.read_text(encoding="utf-8").splitlines():
    if line.startswith("DEEPSEEK_API_KEY="):
        _api_key = line.split("=", 1)[1].strip()
llm = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com")
MODEL = "deepseek-v4-flash"

# 分页安全上限（对齐 Hermes _MCP_LIST_MAX_PAGES = 50）
_MAX_LIST_PAGES = 50

# 工具描述提示注入模式（对齐 Hermes _MCP_INJECTION_PATTERNS 简化版）
_INJECTION_PATTERNS = [
    (re.compile(r"ignore (all |any )?(previous|prior|above|earlier) "
                r"(instructions|prompts|messages)", re.I),
     "试图忽略之前的指令"),
    (re.compile(r"you are now|your new (instructions|role)|system prompt", re.I),
     "试图篡改角色定义"),
    (re.compile(r"disregard|ignore.*(safety|rules|guidelines)", re.I),
     "试图绕过安全规则"),
]


def _scan_mcp_description(server_name: str, tool_name: str, description: str) -> None:
    """扫描工具描述里的提示注入模式，发现可疑内容打警告（对齐 Hermes 573 行）。"""
    if not description:
        return
    findings = [reason for pattern, reason in _INJECTION_PATTERNS
                if pattern.search(description)]
    if findings:
        print(f"[WARN] {server_name}/{tool_name} 描述可疑"
              f"（{'; '.join(findings)}）: {description[:80]}")


class McpProtocol:
    """MCP 协议层（传输无关）：tools/resources/prompts/ping + 分页 + 多模态。

    子类只需实现 send(method, params, notify) -> dict。
    """

    def __init__(self) -> None:
        self._next_id = 0

    # send 一条请求/通知，返回 server 回复的 Json 消息
    async def send(self, method: str, params: dict = None, notify: bool = False) -> dict:
        raise NotImplementedError

    # 分页 -- 限制 Server 一次给的工具个数(MCP方法, 条目在响应的哪个键下（"tools" / "resources" / "prompts"）)
    async def _list_paginated(self, method: str, items_attr: str) -> list:
        """分页拉全（对齐 Hermes _paginate_full_list：跟随 nextCursor，上限 50 页）。"""
        items, cursor = [], None
        for _ in range(_MAX_LIST_PAGES):
            resp = await self.send(method, {"cursor": cursor} if cursor else None)
            result = resp.get("result", {})
            items.extend(result.get(items_attr, []) or [])
            cursor = result.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                break
        return items

    # 列出有的工具
    async def list_tools(self) -> list:
        """问 server 有哪些工具（分页拉全）。"""
        return await self._list_paginated("tools/list", "tools")

    # 调用工具
    async def call_tool(self, name: str, args: dict) -> str:
        """调用工具：文本块直接取，image/audio 等多模态块转摘要。"""
        resp = await self.send("tools/call", {"name": name, "arguments": args})
        if "error" in resp:
            return f"错误: {resp['error'].get('message')}"
        return self._format_content(resp.get("result", {}).get("content", []))

    @staticmethod
    def _format_content(content: list) -> str:
        """结果块 → 文本（resources 内容块无 type 字段，靠 text 键识别）。"""
        parts = []
        for b in content:
            t = b.get("type")
            if t == "text" or "text" in b:
                parts.append(b.get("text", ""))
            elif t == "image":
                data = b.get("data", "")
                parts.append(f"[图片 {b.get('mimeType', 'image/*')}，"
                             f"base64 {len(data)} 字符——视觉模型可分析]")
            elif t == "audio":
                data = b.get("data", "")
                parts.append(f"[音频 {b.get('mimeType', 'audio/*')}，"
                             f"base64 {len(data)} 字符]")
            elif t == "resource":
                parts.append(f"[资源 {b.get('uri', '?')}] {b.get('text', '')}")
        return "\n".join(parts)

    # ── resources（MCP 三件套：数据，按 URI 读取）───────────────
    async def list_resources(self) -> list:
        """列出 server 暴露的数据资源。"""
        return await self._list_paginated("resources/list", "resources")

    async def read_resource(self, uri: str) -> str:
        """按 URI 读取资源内容。"""
        resp = await self.send("resources/read", {"uri": uri})
        if "error" in resp:
            return f"错误: {resp['error'].get('message')}"
        return self._format_content(resp.get("result", {}).get("contents", []))

    # 列出所有的提示词模板
    async def list_prompts(self) -> list:
        """列出 server 的提示模板。"""
        return await self._list_paginated("prompts/list", "prompts")

    async def get_prompt(self, name: str, arguments: dict = None) -> str:
        """获取提示模板（按参数渲染）。"""
        resp = await self.send("prompts/get", {"name": name, "arguments": arguments or {}})
        if "error" in resp:
            return f"错误: {resp['error'].get('message')}"
        msgs = resp.get("result", {}).get("messages", [])
        return "\n".join(m.get("content", {}).get("text", "") for m in msgs)

    # 检查 Server 掉没掉线
    async def ping(self) -> bool:
        """发 ping 等 pong（检查连接是否活着）。"""
        await self.send("ping")
        return True


class StdioMcpClient(McpProtocol):
    """stdio 传输：本地子进程 + recv loop 统一分发（对齐官方 SDK ClientSession）。"""

    def __init__(self, proc) -> None:
        super().__init__()           # _next_id 设为 0
        self.proc = proc
        self._pending: dict = {}     # 请求登记簿 -- id=1 → _pending[1] = Future1
        self._recv_task = None       # 专门读 server 通过管道发给的所有内容

    # ── 接收循环 ────────────────────────────────────────────────
    def start_recv(self, on_notification=None) -> None:
        """启动后台读循环（连接后立刻调用，initialize 响应也走它）。"""
        self._recv_task = asyncio.create_task(self._recv_loop(on_notification))

    async def _recv_loop(self, on_notification) -> None:
        """持续读 server 消息：有 id → 完成对应 future；无 id → 推送回调。"""
        while True:
            try:
                line = await self.proc.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError):
                # 单行超长（如 notion 的大响应 >64KB）：Python 3.11 的
                # readline 会把 LimitOverrunError 转成 ValueError 抛出；
                # read(n) 会等满 n 字节导致死锁 → 直接放弃该连接
                self._fail_pending(RuntimeError("server 响应行超长，连接放弃"))
                break
            if not line:                     # EOF：server 退出了
                self._fail_pending(RuntimeError("server 进程已退出"))
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue                     # banner/日志行，跳过
            msg_id = msg.get("id")
            if msg_id is not None and msg_id in self._pending:
                # 这是我们某个请求的响应 → 完成对应的 future
                fut = self._pending.pop(msg_id)
                if not fut.done():
                    fut.set_result(msg)
            elif msg_id is None:
                # server 主动推送（无 id 的通知）→ 回调
                if on_notification:
                    on_notification(msg)

    # server 挂了时，把所有还在等响应的请求全部判死刑（防止永久卡死）
    def _fail_pending(self, exc: Exception) -> None:
        """让所有等待中的请求失败，避免永久挂起。"""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # ── 发送 ────────────────────────────────────────────────────
    # 消息分两种：请求 -- 带 id，要回复 / 通知 -- 不带 id，不用回复
    async def send(self, method: str, params: dict = None, notify: bool = False) -> dict:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        # 是请求
        if not notify:
            self._next_id += 1
            msg["id"] = self._next_id
            fut = asyncio.get_running_loop().create_future()
            self._pending[msg["id"]] = fut
        # asyncio 管道是二进制的：写入要 encode
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
        await self.proc.stdin.drain()
        # 通知不需要等回复
        if notify:
            return {}                        
        # 等 recv loop 分发结果；30s 超时防止 server 不响应拖死全部
        # （对齐 Hermes mcp_tool.py:2206 的 wait_for(timeout=30.0)）
        return await asyncio.wait_for(fut, timeout=30)

    # 关闭子进程
    def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self.proc.returncode is None:
            self.proc.terminate()


class HttpMcpClient(McpProtocol):
    """远程传输：streamable HTTP（aiohttp POST + SSE 响应解析）。

    对齐 Hermes 的远程 server 支持（figma/linear 那种 URL 连接）。
    """

    def __init__(self, url: str, headers: dict = None, timeout: float = 30.0) -> None:
        super().__init__()
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    async def send(self, method: str, params: dict = None, notify: bool = False) -> dict:
        import aiohttp
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._next_id += 1
            msg["id"] = self._next_id
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.url, json=msg, headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "text/event-stream" in ctype:
                    # SSE 流：逐行找 data: 里的 JSON 消息（远程 server 常用）
                    return self._parse_sse(await resp.text(), msg.get("id"))
                return await resp.json()

    @staticmethod
    def _parse_sse(text: str, req_id: int) -> dict:
        """解析 SSE 流：找 data: 行里的 JSON，匹配请求 id。"""
        for line in text.splitlines():
            if line.startswith("data:"):
                try:
                    msg = json.loads(line[5:].strip())
                    if req_id is None or msg.get("id") == req_id:
                        return msg
                except json.JSONDecodeError:
                    continue
        return {"error": {"message": "SSE 流中未找到响应"}}

    def close(self) -> None:
        """远程无子进程可关（每次请求独立连接）。"""


class MCPManager:
    """MCP 连接管理（懒加载，对齐 Hermes 的 lazy start on first use）。

    启动：有 schema 缓存的 server 只注册工具名（进程不启动）；
          无缓存的并发连接（首次总要拉一次工具清单建缓存）。
    调用：_ensure_server 现场检查——活着直接用，死了现场重连，
          冷却中/连接中本次调用失败（不硬等）。
    """

    def __init__(self, config_path: Path) -> None:
        self.servers = json.loads(config_path.read_text(encoding="utf-8"))["servers"]
        self.clients: dict = {}    # server名 -> McpProtocol（stdio 或 http）
        self.tools: dict = {}      # 前缀工具名 -> (server名, 工具定义)
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        # ── 懒加载状态（对齐 Hermes）──
        self._connecting: set = set()          # 正在连接的 server（去重）
        self._cooldown_until: dict = {}        # server名 -> 冷却到期时间
        self._backoff: dict = {}               # server名 -> 失败次数
        self._schema_cache: dict = self._load_schema_cache()  # 读出来的工具清单

    # ── schema 缓存（磁盘持久化，跨进程复用）────────────────────
    def _load_schema_cache(self) -> dict:
        """读磁盘 schema 缓存（对齐 Hermes mcp_schema_cache）。"""
        if SCHEMA_CACHE_FILE.exists():
            try:
                return json.loads(SCHEMA_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_schema_cache(self) -> None:
        """写磁盘 schema 缓存。"""
        SCHEMA_CACHE_FILE.write_text(
            json.dumps(self._schema_cache, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # ── 启动 / 关闭 ─────────────────────────────────────────────
    def start(self) -> None:
        """启动后台事件循环线程 + 懒发现（阻塞等无缓存 server 连接完）。"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._discover_all(), self._loop).result()

    def _run_loop(self) -> None:
        """后台线程入口：事件循环常驻（run_forever，直到 stop）。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self) -> None:
        """关闭：关已连接的 client、停事件循环。"""
        for client in self.clients.values():
            client.close()
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ── 发现（启动时一次）───────────────────────────────────────
    async def _discover_all(self) -> None:
        """启动发现：有缓存的懒注册名字；无缓存的并发连接（gather）。"""
        need_connect = []
        for name, cfg in self.servers.items():
            fp = self._server_fingerprint(cfg)
            if fp in self._schema_cache:
                # 懒加载：工具名已缓存 → 只注册名字，进程等首次调用
                self._register_tools(name, self._schema_cache[fp])
                print(f"[MCP] {name} server 懒加载"
                      f"（工具名已注册，进程首次调用时启动）")
            else:
                need_connect.append((name, cfg))
        if need_connect:
            results = await asyncio.gather(
                *(self._connect_one(n, c) for n, c in need_connect),
                return_exceptions=True,
            )
            for (name, _cfg), result in zip(need_connect, results):
                if isinstance(result, BaseException):
                    self._server_connect_errors[name] = str(result)
                    print(f"[MCP] {name} server 连接失败（{result}）——调用时再试")
                else:
                    print(f"[MCP] 已连接 {name} server")

    @staticmethod
    def _server_fingerprint(cfg: dict) -> str:
        """单个 server 的配置指纹：command/args/env/url 变了指纹就变。"""
        raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()

    def _register_tools(self, name: str, tools: list) -> None:
        """工具清单 → 注册表（前缀防冲突 + read_resource + 安全扫描）。"""
        for t in tools:
            _scan_mcp_description(name, t["name"], t.get("description", ""))
            self.tools[f"{name}_{t['name']}"] = (name, t)
        # 每个 server 注册一个 read_resource 工具（对齐 Hermes 的封装方式）
        self.tools[f"{name}_read_resource"] = (
            name, {"name": "read_resource",
                   "description": f"按 URI 读取 {name} server 的数据资源",
                   "inputSchema": {"type": "object",
                                   "properties": {"uri": {"type": "string"}},
                                   "required": ["uri"]}})

    async def _connect_one(self, name: str, cfg: dict):
        """连接单个 server：按传输类型建 client + 握手 + 拉工具 + 注册 + 写缓存。"""
        # 子进程环境 = 当前环境 + server 配置的 env（如 API key）
        env = {**os.environ}
        env.update({k: v for k, v in (cfg.get("env") or {}).items() if v})

        if cfg.get("url"):
            # 远程传输：HTTP（figma/linear 那种 URL server）
            client: McpProtocol = HttpMcpClient(
                cfg["url"], headers={"Authorization": f"Bearer {env.get('MCP_TOKEN', '')}"}
                if env.get("MCP_TOKEN") else None)
        else:
            # stdio 传输：本地子进程 + recv loop
            proc = await asyncio.create_subprocess_exec(
                cfg["command"], *cfg.get("args", []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
            )
            stdio_client = StdioMcpClient(proc)
            stdio_client.start_recv(on_notification=lambda m: print(
                f"[MCP] {name} server 推送: {json.dumps(m, ensure_ascii=False)[:100]}"))
            client = stdio_client

        await client.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "llm-mcp-demo", "version": "0.1.0"},
        })
        await client.send("notifications/initialized", notify=True)
        tools = await client.list_tools()
        # 注册 + 写磁盘缓存（下次启动可懒加载）
        self._register_tools(name, tools)
        self._schema_cache[self._server_fingerprint(cfg)] = tools
        self._save_schema_cache()
        self.clients[name] = client

    # ── 调用时懒加载（对齐 Hermes lazy_start_server）────────────
    async def _is_alive(self, client) -> bool:
        """检查连接是否活着：stdio 查子进程；远程 ping。"""
        if isinstance(client, StdioMcpClient):
            return client.proc.returncode is None
        try:
            await asyncio.wait_for(client.ping(), timeout=5)
            return True
        except Exception:
            return False

    async def _ensure_server(self, name: str) -> bool:
        """调用时确保 server 可用（对齐 Hermes lazy_start_server 4781 行）：
        活着 → True；死了 → 现场重连；冷却中/连接中 → False。"""
        client = self.clients.get(name)
        if client is not None and await self._is_alive(client):
            return True
        now = asyncio.get_running_loop().time()
        if name in self._cooldown_until and now < self._cooldown_until[name]:
            return False                       # 冷却中：不反复重试
        if name in self._connecting:
            return False                       # 已有连接任务在跑（去重）
        cfg = self.servers.get(name)
        if cfg is None:
            return False
        self._connecting.add(name)
        try:
            await self._connect_one(name, cfg)
            self._backoff.pop(name, None)
            self._server_connect_errors.pop(name, None)
            print(f"[MCP] {name} server 已重新连接")
            return True
        except Exception as exc:
            self._server_connect_errors[name] = str(exc)
            delay = min(2 ** self._backoff.get(name, 0), 30)
            self._backoff[name] = self._backoff.get(name, 0) + 1
            self._cooldown_until[name] = now + delay
            print(f"[MCP] {name} 重连失败（{exc}），{delay}s 冷却")
            return False
        finally:
            self._connecting.discard(name)

    # ── 供 LLM 循环调用（主线程 → 后台循环）────────────────────
    def tool_schemas(self) -> list:
        """注册表 → OpenAI tools 格式（LLM 每轮看到的就是这个）。"""
        result = []
        for prefixed_name, (server_name, t) in self.tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": prefixed_name,
                    "description": f"[{server_name}] {t['description']}",
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })
        return result

    def call(self, tool_name: str, args: dict) -> str:
        """主线程调用工具：投递到后台事件循环执行，阻塞等结果返回。"""
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(tool_name, args), self._loop)
        return future.result()

    async def _call_async(self, tool_name: str, args: dict) -> str:
        """后台执行：先懒加载确保 server 可用，再路由调用。"""
        server_name, _, real_name = tool_name.partition("_")
        if not await self._ensure_server(server_name):
            return f"错误: server {server_name} 不可用（掉线/冷却中/正在连接）"
        client = self.clients.get(server_name)
        if real_name == "read_resource":
            # 特殊工具：读 server 的数据资源（MCP 三件套之一）
            return await client.read_resource(args.get("uri", ""))
        return await client.call_tool(real_name, args)


def main() -> None:
    """演示主流程：懒加载 MCP 连接 + LLM 自主调工具。"""
    question = sys.argv[1] if len(sys.argv) > 1 else "看看 zsrc 目录下有哪些 .py 文件"

    manager = MCPManager(MCP_CFG)
    manager.start()
    print(f"[MCP] 共注册 {len(manager.tools)} 个工具（懒加载："
          f"{len(manager.clients)} 个已连接，其余首次调用时启动）\n")

    messages = [
        {"role": "system", "content": "你是一个助手。需要信息时调用工具，然后基于工具结果回答。"},
        {"role": "user", "content": question},
    ]
    schemas = manager.tool_schemas()

    for _round in range(8):  # 最多 8 轮工具循环
        resp = llm.chat.completions.create(model=MODEL, messages=messages, tools=schemas)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(f"[LLM] {msg.content}")
            break
        # LLM 决定调工具 → 通过 manager 执行（首次调用会现场启动 server）
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name, args = tc.function.name, json.loads(tc.function.arguments or "{}")
            print(f"[LLM] 决定调用工具: {name}({json.dumps(args, ensure_ascii=False)})")
            result = manager.call(name, args)
            print(f"[MCP] 执行结果（前 200 字）: {result[:200]}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result[:2000]})

    manager.shutdown()
    print("[MCP] 连接已关闭")


if __name__ == "__main__":
    main()
