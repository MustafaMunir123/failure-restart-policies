"""Phase 0 wiring check on Kaggle T4 — proves the full Thinkingbox stack runs
with a local vLLM Qwen3 (thinking off) end to end.

Everything heavy lives in /tmp (ephemeral). Only artifacts go to /kaggle/working.
Always writes /kaggle/working/result.json (success or traceback).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
BASE = "/tmp/tb"
os.makedirs(BASE, exist_ok=True)
VENV = f"{BASE}/venv"
UV = shutil.which("uv") or "/usr/local/bin/uv"  # resolved AFTER install step
LOGS = f"{WORK}/logs"
os.makedirs(LOGS, exist_ok=True)

RESULT = {"steps": {}, "ok": False}


def log_step(name, ok, detail=""):
    RESULT["steps"][name] = {"ok": bool(ok), "detail": str(detail)[:2000]}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:500]}", flush=True)


def run(cmd, name, cwd=None, env=None, timeout=3600, check=True):
    print(f"\n=== {name} ===\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    e = dict(os.environ, **(env or {}))
    with open(f"{LOGS}/{name.replace(' ', '_').replace('/', '_')}.log", "wb") as lf:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=e,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    log_step(name, p.returncode == 0, f"rc={p.returncode}")
    if check and p.returncode != 0:
        raise RuntimeError(f"{name} failed rc={p.returncode} — see logs/{name}.log")
    return p.returncode == 0


def wait_port(port, proc, name, timeout=900):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} died rc={proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(5)
    raise RuntimeError(f"{name} port {port} never opened")


try:
    # ---------- 1. uv + python 3.12 ----------
    run("curl -LsSf https://astral.sh/uv/install.sh | sh", "install uv")
    os.environ["PATH"] = "/root/.local/bin:/usr/local/bin:" + os.environ["PATH"]
    UV = shutil.which("uv") or "/usr/local/bin/uv"
    log_step("resolve uv", shutil.which("uv") is not None, UV)

    # ---------- 2. clone + pin ----------
    run("git clone --depth 50 https://github.com/microsoft/thinkingbox.git thinkingbox",
        "clone framework", cwd=BASE)
    run("git clone https://github.com/microsoft/thinkingbox-data.git thinkingbox-data",
        "clone data", cwd=BASE)
    run("git checkout thinkingbox-bench-v1.0", "pin data tag",
        cwd=f"{BASE}/thinkingbox-data")
    for repo in ("thinkingbox", "thinkingbox-data"):
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=f"{BASE}/{repo}",
                             capture_output=True, text=True).stdout.strip()
        RESULT[f"commit_{repo}"] = sha

    # ---------- 3. framework venv (python 3.12) ----------
    env = dict(os.environ,
               UV_PROJECT_ENVIRONMENT=VENV,
               VIRTUAL_ENV=VENV,
               PATH=f"{VENV}/bin:" + os.environ["PATH"],
               THINKINGBOX_DATA=f"{BASE}/thinkingbox-data",
               TB_MCP_START_SERVERS_FILE=f"{BASE}/thinkingbox-data/servers/servers.yaml",
               TYPESENSE_API_KEY="Fake")
    run([UV, "venv", "--python", "3.12", VENV], "create py3.12 venv")
    run([UV, "sync", "--group", "dev"], "uv sync framework", cwd=f"{BASE}/thinkingbox",
        env=env, timeout=5400)
    run([UV, "pip", "install", "--python", f"{VENV}/bin/python",
         "--config-settings", "editable-mode=compat",
         "-e", f"{BASE}/thinkingbox-data/servers/tb_business_ops_servers_202606"],
        "install MCP server pkg", env=env, timeout=1800)

    # ---------- 4. typesense (use framework's own installer) ----------
    run(["bash", "scripts/install_typesense.sh"], "install typesense",
        cwd=f"{BASE}/thinkingbox", env=env, timeout=900)
    # NOTE: typesense 30.1 has no --version flag; it prints the banner then exits 1
    # demanding --data-dir. So: non-fatal check, PASS if banner appears.
    rc = run("typesense-server --version", "check typesense", env=env, check=False)
    banner = open(f"{LOGS}/check_typesense.log").read()
    log_step("verify typesense binary", "Typesense" in banner,
             banner.splitlines()[0] if banner else "empty")

    # ---------- 5. vLLM serving Qwen3-4B ----------
    VVENV = f"{BASE}/vllm-venv"
    run([UV, "venv", "--python", "3.12", VVENV], "create vllm venv")
    run([UV, "pip", "install", "--python", f"{VVENV}/bin/python", "vllm"],
        "install vllm", timeout=7200)
    vllm_log = open(f"{LOGS}/vllm_serve.log", "wb")
    vllm_proc = subprocess.Popen(
        [f"{VVENV}/bin/vllm", "serve", "Qwen/Qwen3-4B",
         "--served-model-name", "Qwen3-4B", "--port", "8000",
         "--dtype", "float16", "--max-model-len", "16384",
         "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
         "--gpu-memory-utilization", "0.85"],
        stdout=vllm_log, stderr=subprocess.STDOUT)
    wait_port(8000, vllm_proc, "vllm")

    # ---------- 6. start typesense + session proxy ----------
    ts_data = f"{BASE}/typesense-data"; os.makedirs(ts_data, exist_ok=True)
    ts_log = open(f"{LOGS}/typesense.log", "wb")
    ts_proc = subprocess.Popen(
        ["typesense-server", "--data-dir", ts_data, "--api-key", "Fake",
         "--enable-cors"], env=env, stdout=ts_log, stderr=subprocess.STDOUT)
    wait_port(8108, ts_proc, "typesense")

    proxy_log = open(f"{LOGS}/session_proxy.log", "wb")
    proxy_proc = subprocess.Popen(
        [UV, "run", "tb", "mcp-start",
         "--servers", f"{BASE}/thinkingbox-data/servers/servers.yaml"],
        cwd=f"{BASE}/thinkingbox", env=env, stdout=proxy_log, stderr=subprocess.STDOUT)
    time.sleep(30)  # give MCP servers time to register; proxy has no fixed port contract here
    log_step("start session proxy", proxy_proc.poll() is None,
             f"pid={proxy_proc.pid}")

    # ---------- 7. write LLM config (schema per pinned config_types.py) ----------
    cfg = """mcp_proxy:
  endpoint_url: "http://127.0.0.1:7111"
  timeout: 300.0

orchestrator:
  type: thinkingbox
  agent_model:
    type: aoai
    deployment: "Qwen3-4B"
    endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
    credential:
      type: api-key
      api_key: "EMPTY"
    is_reasoning: false
    reasoning_source: none
    temperature: 0.7
    max_completion_tokens: 4096
    timeout: 120.0

judge_model:
  type: aoai
  deployment: "Qwen3-4B"
  endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
  credential:
    type: api-key
    api_key: "EMPTY"
  is_reasoning: false
  reasoning_source: none
  temperature: 0.0
  seed: 42
  max_completion_tokens: 128
  timeout: 60.0

judge_type: "legacy"

user_model:
  type: aoai
  deployment: "Qwen3-4B"
  endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
  credential:
    type: api-key
    api_key: "EMPTY"
  is_reasoning: false
  reasoning_source: none
  temperature: 0.3
  seed: 42
  max_completion_tokens: 512
  timeout: 60.0
"""
    cfg_path = f"{BASE}/wiring_config.yaml"
    open(cfg_path, "w").write(cfg)
    # NOTE: field names verified against config_types.py on the pinned commit at runtime;
    # if tb rejects the config, the error will say which key is wrong -> see logs.

    # ---------- 8. wiring check ----------
    run([UV, "run", "tb", "infer",
         "-c", cfg_path,
         "--dataset", f"{BASE}/thinkingbox-data/dataset", "--agent", "think",
         "--name", "banking.py:test_get_balance_savings",
         "--output", f"{WORK}/wiring_check_output.yaml"],
        "tb infer wiring check", cwd=f"{BASE}/thinkingbox", env=env, timeout=1800)
    run([UV, "run", "tb", "pp", f"{WORK}/wiring_check_output.yaml"],
        "tb pretty-print", cwd=f"{BASE}/thinkingbox", env=env)

    RESULT["ok"] = True

except Exception as e:
    RESULT["error"] = "".join(traceback.format_exc())[-4000:]
    print("FATAL:", e, flush=True)
finally:
    for name in ("vllm_proc", "ts_proc", "proxy_proc"):
        proc = locals().get(name)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
    with open(f"{WORK}/result.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print("\nRESULT:", json.dumps({k: v for k, v in RESULT.items() if k != "steps"},
                                  indent=2), flush=True)
