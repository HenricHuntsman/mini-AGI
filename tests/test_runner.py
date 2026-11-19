import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerEvidenceTests(unittest.TestCase):
    def test_timeout_stops_child_and_preserves_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp) / 'test_child.py'
            source.write_text('import subprocess,sys,time\n'
                              'subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                              'print("child started", flush=True)\n'
                              'time.sleep(60)\n')
            report = Path(tmp) / 'results'
            start = time.monotonic()
            result = subprocess.run([
                sys.executable, str(ROOT / 'tools/run_test_suite.py'),
                '--timeout', '1', '--report-dir', str(report), str(source)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertLess(time.monotonic() - start, 10)
            summary = json.loads((report / 'summary.json').read_text())
            self.assertEqual(summary['files'][0]['exit_code'], 124)
            self.assertIn('child started', (report / 'test_child.py.log').read_text())


if __name__ == '__main__':
    unittest.main()
