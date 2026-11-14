"""Run unittest entry points and executable regressions with offline networking."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import json
import platform
import signal
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("files", nargs="*", help="Optional test file paths")
    parser.add_argument("--report-dir", help="Save every file log and a JSON outcome summary")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    root = Path(__file__).resolve().parent.parent
    paths = [root / name for name in args.files] if args.files else sorted(
        (root / "tests").glob("test_*.py"))
    if not paths or any(not path.is_file() for path in paths):
        parser.error("expected at least one existing test file")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    # Tests use local fixtures. Do not route them through inherited proxies or
    # let a developer's notification credentials enter a child process.
    for key in list(env):
        if key.lower().endswith("_proxy") or key in (
                "MINAGI_TELEGRAM_TOKEN", "MINAGI_TELEGRAM_CHAT"):
            env.pop(key)
    # A fresh process prevents a script's sys.exit or torch process globals
    # from ending discovery or changing another regression's environment.
    runner = '''import ipaddress,runpy,socket,sys
def local(host):
    if isinstance(host, bytes): host = host.decode()
    if host == "localhost": return True
    try: return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError): return False
original_lookup = socket.getaddrinfo
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
def lookup(host, *args, **kwargs):
    if not local(host): raise OSError("Offline test blocked external hostname")
    return original_lookup(host, *args, **kwargs)
def connect(self, address):
    if isinstance(address, tuple) and not local(address[0]):
        raise OSError("Offline test blocked external connection")
    return original_connect(self, address)
def connect_ex(self, address):
    if isinstance(address, tuple) and not local(address[0]):
        raise OSError("Offline test blocked external connection")
    return original_connect_ex(self, address)
socket.getaddrinfo = lookup
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''

    def run(path):
        process = subprocess.Popen(
            [sys.executable, "-c", runner, str(path)], cwd=root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=(os.name == "posix"))
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
            return path, process.returncode, stdout + stderr
        except subprocess.TimeoutExpired:
            # A test may spawn a trainer. Killing only the test leaves that
            # trainer running and holding the captured pipes open indefinitely.
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
            return path, 124, (stdout + stderr +
                              f"\nTimed out after {args.timeout:g} seconds")

    report = Path(args.report_dir) if args.report_dir else None
    if report:
        report.mkdir(parents=True, exist_ok=True)
    records = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, code, output in pool.map(run, paths):
            print(f"{'FAIL' if code else 'PASS'} {path.name} (exit {code})", flush=True)
            records.append({"file": str(path.relative_to(root)), "exit_code": code,
                            "outcome": "passed" if code == 0 else "failed"})
            if report:
                (report / (path.name + ".log")).write_text(output, encoding="utf-8")
            if code:
                failed += 1
                print(output, flush=True)
    print(f"Files: {len(paths)}; passed: {len(paths) - failed}; failed: {failed}")
    if report:
        (report / "summary.json").write_text(json.dumps({
            "python": platform.python_version(), "platform": platform.platform(),
            "coverage_unit": "test file process exit; inspect logs for test skips",
            "files": records, "passed": len(paths) - failed, "failed": failed,
        }, indent=2) + "\n", encoding="utf-8")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
