import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OnboardingSmokeTests(unittest.TestCase):
    def test_eval_cli_advertises_json_output_artifact(self):
        result = subprocess.run(
            [sys.executable, "eval_perplexity.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--output_json", result.stdout)
        self.assertIn("--run_id", result.stdout)

    def test_smoke_script_runs_from_repository_root(self):
        smoke = (ROOT / "scripts" / "smoke_test.sh").read_text(encoding="utf-8")

        self.assertIn('REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."', smoke)
        self.assertIn('cd "$REPO_ROOT"', smoke)


if __name__ == "__main__":
    unittest.main()
