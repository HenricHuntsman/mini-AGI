"""Run unittest entry points and executable regressions with offline networking."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("files", nargs="*", help="Optional test file paths")
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
        try:
            result = subprocess.run(
                [sys.executable, "-c", runner, str(path)], cwd=root, env=env,
                capture_output=True, text=True, timeout=args.timeout)
            return path, result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return path, 124, f"Timed out after {args.timeout:g} seconds"

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, code, output in pool.map(run, paths):
            print(f"{'FAIL' if code else 'PASS'} {path.name} (exit {code})", flush=True)
            if code:
                failed += 1
                print(output, flush=True)
    print(f"Files: {len(paths)}; passed: {len(paths) - failed}; failed: {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
