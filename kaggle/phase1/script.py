"""Phase 1 smoke matrix on Kaggle T4 — feasibility + harness validation.

For each size (1.7B, 4B, 8B-AWQ): serve with vLLM, run 5 benchmark tasks x 2
repetitions thinking-off, record wall time and completion per run.
Then one diagnostic: Qwen3-4B WITH thinking (is_reasoning: true).

Sequential serving (one model at a time). All artifacts -> /kaggle/working/.
Always writes /kaggle/working/result.json.
"""
import json
import os
import shutil as sh
import subprocess
import time
import traceback

WORK = "/kaggle/working"
BASE = "/tmp/tb"
os.makedirs(BASE, exist_ok=True)
VENV = f"{BASE}/venv"
LOGS = f"{WORK}/logs"
os.makedirs(LOGS, exist_ok=True)
UV = sh.which("uv") or "/usr/local/bin/uv"

RESULT = {"runs": [], "models": {}, "ok": False}


def log_step(name, ok, detail=""):
    RESULT.setdefault("steps", {})[name] = {"ok": bool(ok), "detail": str(detail)[:1500]}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:400]}", flush=True)


def run(cmd, name, cwd=None, env=None, timeout=3600, check=True):
    print(f"\n=== {name} ===", flush=True)
    e = dict(os.environ, **(env or {}))
    with open(f"{LOGS}/{name.replace(' ', '_').replace('/', '_')}.log", "wb") as lf:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=e,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    log_step(name, p.returncode == 0, f"rc={p.returncode}")
    if check and p.returncode != 0:
        raise RuntimeError(f"{name} failed rc={p.returncode}")
    return p.returncode == 0


def wait_port(port, proc, name, timeout=1200):
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


def write_config(path, model_name, thinking=False):
    reason = ("is_reasoning: true\n    reasoning_source: content"
              if thinking else
              "is_reasoning: false\n    reasoning_source: none")
    def block(dep, temp, max_tok, seed=None):
        s = f"\n  seed: {seed}" if seed is not None else ""
        return f"""    type: aoai
    deployment: "{dep}"
    endpoint_url: "http://127.0.0.1:8000/v1/chat/completions"
    credential:
      type: api-key
      api_key: "EMPTY"
    {reason}
    temperature: {temp}{s}
    max_completion_tokens: {max_tok}
    timeout: 120.0"""
    cfg = f"""mcp_proxy:
  endpoint_url: "http://127.0.0.1:7111"
  timeout: 300.0

orchestrator:
  type: thinkingbox
  agent_model:
{block(model_name, 0.7, 4096)}

judge_model:
{block(model_name, 0.0, 128, seed=42)}

judge_type: "legacy"

user_model:
{block(model_name, 0.3, 512, seed=42)}
"""
    open(path, "w").write(cfg)


PROCS = {}
try:
    # ---------- setup (identical to proven phase-0 chain) ----------
    run("curl -LsSf https://astral.sh/uv/install.sh | sh", "install uv")
    os.environ["PATH"] = "/root/.local/bin:/usr/local/bin:" + os.environ["PATH"]
    UV = sh.which("uv") or "/usr/local/bin/uv"
    run(f"git clone --depth 50 https://github.com/microsoft/thinkingbox.git thinkingbox",
        "clone framework", cwd=BASE)
    run("git clone https://github.com/microsoft/thinkingbox-data.git thinkingbox-data",
        "clone data", cwd=BASE)
    run("git checkout thinkingbox-bench-v1.0", "pin data tag", cwd=f"{BASE}/thinkingbox-data")
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=VENV, VIRTUAL_ENV=VENV,
               PATH=f"{VENV}/bin:" + os.environ["PATH"],
               THINKINGBOX_DATA=f"{BASE}/thinkingbox-data",
               TB_MCP_START_SERVERS_FILE=f"{BASE}/thinkingbox-data/servers/servers.yaml",
               TYPESENSE_API_KEY="Fake")
    run([UV, "venv", "--python", "3.12", VENV], "create py3.12 venv")
    run([UV, "sync", "--group", "dev"], "uv sync framework",
        cwd=f"{BASE}/thinkingbox", env=env, timeout=5400)
    run([UV, "pip", "install", "--python", f"{VENV}/bin/python",
         "--config-settings", "editable-mode=compat",
         "-e", f"{BASE}/thinkingbox-data/servers/tb_business_ops_servers_202606"],
        "install MCP server pkg", env=env, timeout=1800)
    run(["bash", "scripts/install_typesense.sh"], "install typesense",
        cwd=f"{BASE}/thinkingbox", env=env, timeout=900)

    # ---------- services ----------
    VVENV = f"{BASE}/vllm-venv"
    run([UV, "venv", "--python", "3.12", VVENV], "create vllm venv")
    run([UV, "pip", "install", "--python", f"{VVENV}/bin/python", "vllm"],
        "install vllm", timeout=7200)

    ts_data = f"{BASE}/typesense-data"; os.makedirs(ts_data, exist_ok=True)
    PROCS["ts"] = subprocess.Popen(
        ["typesense-server", "--data-dir", ts_data, "--api-key", "Fake"],
        env=env, stdout=open(f"{LOGS}/typesense.log", "wb"),
        stderr=subprocess.STDOUT)
    wait_port(8108, PROCS["ts"], "typesense")
    PROCS["proxy"] = subprocess.Popen(
        [UV, "run", "tb", "mcp-start",
         "--servers", f"{BASE}/thinkingbox-data/servers/servers.yaml"],
        cwd=f"{BASE}/thinkingbox", env=env,
        stdout=open(f"{LOGS}/session_proxy.log", "wb"), stderr=subprocess.STDOUT)
    time.sleep(30)
    log_step("start session proxy", PROCS["proxy"].poll() is None)

    # ---------- pick 5 smoke tasks from canonical testlist (deterministic) ----------
    sel_code = (
        "import yaml,json\n"
        f"tl=yaml.safe_load(open('{BASE}/thinkingbox-data/releases/"
        "thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml'))\n"
        "names=[]\n"
        "if isinstance(tl,dict):\n"
        "    tl=tl.get('tests') or tl.get('tasks') or list(tl.values())\n"
        "for e in tl:\n"
        "    n=e if isinstance(e,str) else (e.get('name') or e.get('test') or '')\n"
        "    if isinstance(n,str) and '.' in n and ':' in n: names.append(n)\n"
        "names=sorted(set(names))\n"
        "# first 5 across distinct source files for domain spread\n"
        "picked,seen=[],set()\n"
        "for n in names:\n"
        "    f=n.split(':')[0]\n"
        "    if f not in seen:\n"
        "        picked.append(n); seen.add(f)\n"
        "    if len(picked)==5: break\n"
        "json.dump(picked, open('/tmp/tb/smoke_tasks.json','w'))\n"
        "print(picked)\n")
    run([f"{VENV}/bin/python", "-c", sel_code], "select smoke tasks")

    TASKS = json.load(open("/tmp/tb/smoke_tasks.json"))
    if len(TASKS) < 3:  # fallback to known-good names
        TASKS = ["banking.py:test_get_balance_savings",
                 "mcs_defaults.py:test_mcs_defaults_easy"]
    RESULT["smoke_tasks"] = TASKS
    print("SMOKE TASKS:", TASKS, flush=True)

    # ---------- smoke matrix: sequential models ----------
    MODELS = [
        ("Qwen3-1.7B", "Qwen/Qwen3-1.7B", None),
        ("Qwen3-4B", "Qwen/Qwen3-4B", None),
        ("Qwen3-8B-AWQ", "Qwen/Qwen3-8B-AWQ", "--quantization awq"),
    ]
    for label, hf_id, extra in MODELS:
        t_model = time.time()
        PROCS["vllm"] = subprocess.Popen(
            [f"{VVENV}/bin/vllm", "serve", hf_id,
             "--served-model-name", label, "--port", "8000",
             "--dtype", "float16", "--max-model-len", "16384",
             "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
             "--gpu-memory-utilization", "0.90"]
            + (extra.split() if extra else []),
            stdout=open(f"{LOGS}/vllm_{label}.log", "wb"), stderr=subprocess.STDOUT)
        try:
            wait_port(8000, PROCS["vllm"], f"vllm-{label}")
            cfg = f"{BASE}/cfg_{label}.yaml"
            write_config(cfg, label, thinking=False)
            mstats = {"model": label, "load_s": round(time.time() - t_model, 1),
                      "runs": []}
            for task in TASKS:
                for rep in range(2):
                    t0 = time.time()
                    out = f"{WORK}/smoke_{label}_{task.replace(':', '_').replace('/', '_')}_r{rep}.yaml"
                    rc = run([UV, "run", "tb", "infer", "-c", cfg,
                              "--dataset", f"{BASE}/thinkingbox-data/dataset",
                              "--agent", "think", "--name", task, "--output", out],
                             f"infer {label} {task} r{rep}",
                             cwd=f"{BASE}/thinkingbox", env=env,
                             timeout=1500, check=False)
                    mstats["runs"].append({
                        "task": task, "rep": rep,
                        "rc": rc, "wall_s": round(time.time() - t0, 1),
                        "output_written": os.path.exists(out)})
                    with open(f"{WORK}/smoke_results.jsonl", "a") as f:
                        f.write(json.dumps(mstats["runs"][-1] | {"model": label}) + "\n")
            ok_runs = sum(1 for r in mstats["runs"] if r["rc"] == 0)
            walls = [r["wall_s"] for r in mstats["runs"]]
            mstats.update(completed=ok_runs, total=len(mstats["runs"]),
                          avg_wall_s=round(sum(walls) / len(walls), 1) if walls else None)
            RESULT["models"][label] = mstats
        finally:
            PROCS["vllm"].terminate(); PROCS["vllm"].wait(timeout=120)
            time.sleep(10)

    # ---------- diagnostic: Qwen3-4B WITH thinking ----------
    try:
        t_model = time.time()
        PROCS["vllm"] = subprocess.Popen(
            [f"{VVENV}/bin/vllm", "serve", "Qwen/Qwen3-4B",
             "--served-model-name", "Qwen3-4B-think", "--port", "8000",
             "--dtype", "float16", "--max-model-len", "16384",
             "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
             "--reasoning-parser", "qwen3", "--gpu-memory-utilization", "0.90"],
            stdout=open(f"{LOGS}/vllm_4B_think.log", "wb"), stderr=subprocess.STDOUT)
        wait_port(8000, PROCS["vllm"], "vllm-diagnostic")
        cfg = f"{BASE}/cfg_think.yaml"
        write_config(cfg, "Qwen3-4B-think", thinking=True)
        dstats = []
        for task in TASKS[:2]:
            t0 = time.time()
            out = f"{WORK}/diag_think_{task.replace(':', '_')}.yaml"
            rc = run([UV, "run", "tb", "infer", "-c", cfg,
                      "--dataset", f"{BASE}/thinkingbox-data/dataset",
                      "--agent", "think", "--name", task, "--output", out],
                     f"diagnostic think {task}", cwd=f"{BASE}/thinkingbox",
                     env=env, timeout=1800, check=False)
            dstats.append({"task": task, "rc": rc,
                           "wall_s": round(time.time() - t0, 1)})
        RESULT["diagnostic_thinking_on"] = dstats
    finally:
        PROCS["vllm"].terminate(); PROCS["vllm"].wait(timeout=120)

    RESULT["ok"] = True

except Exception as e:
    RESULT["error"] = "".join(traceback.format_exc())[-4000:]
    print("FATAL:", e, flush=True)
finally:
    for p in PROCS.values():
        try:
            p.terminate()
        except Exception:
            pass
    with open(f"{WORK}/result.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    summary = {k: v for k, v in RESULT.items() if k != "runs"}
    print("\nSUMMARY:", json.dumps(summary, indent=2)[:3000], flush=True)
