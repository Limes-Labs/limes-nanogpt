import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RLToySmokeTests(unittest.TestCase):
    def test_toy_reward_prefers_exact_formatted_answer(self):
        from rl.ppo_toy import score_completion

        exact = score_completion("AB", target="AB", ended=True)
        wrong = score_completion("BA", target="AB", ended=True)
        unformatted = score_completion("AB", target="AB", ended=False)

        self.assertGreater(exact, wrong)
        self.assertGreater(exact, unformatted)

    def test_ppo_toy_writes_reproducible_run_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ppo_toy.json"
            subprocess.run(
                [
                    sys.executable,
                    "rl/ppo_toy.py",
                    "--steps=3",
                    "--batch_size=8",
                    "--seed=7",
                    f"--output_json={output}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(artifact["schema_version"], "limes-rl-toy-v0")
        self.assertEqual(artifact["config"]["target"], "AB")
        self.assertEqual(len(artifact["history"]), 3)
        self.assertIn("avg_reward", artifact["final"])
        self.assertIn("approx_kl_to_reference", artifact["final"])

    def test_grpo_toy_writes_group_relative_run_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "grpo_toy.json"
            subprocess.run(
                [
                    sys.executable,
                    "rl/grpo_toy.py",
                    "--steps=3",
                    "--groups=4",
                    "--group_size=4",
                    "--seed=7",
                    f"--output_json={output}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(artifact["schema_version"], "limes-grpo-toy-v0")
        self.assertEqual(artifact["config"]["target"], "AB")
        self.assertEqual(artifact["config"]["groups"], 4)
        self.assertEqual(artifact["config"]["group_size"], 4)
        self.assertFalse(artifact["config"]["uses_value_head"])
        self.assertEqual(len(artifact["history"]), 3)
        self.assertIn("avg_group_reward_std", artifact["final"])
        self.assertIn("approx_kl_to_reference", artifact["final"])

    def test_rl_smoke_script_runs_from_repository_root(self):
        smoke = (ROOT / "scripts" / "rl_toy_smoke.sh").read_text(encoding="utf-8")

        self.assertIn('REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."', smoke)
        self.assertIn('cd "$REPO_ROOT"', smoke)
        self.assertIn("rl/ppo_toy.py", smoke)
        self.assertIn("rl/grpo_toy.py", smoke)


class EfficiencyScoreTests(unittest.TestCase):
    def test_bpb_from_nats_is_tokenizer_agnostic(self):
        from scripts.efficiency_score import bits_per_byte_from_nats

        bpb = bits_per_byte_from_nats(
            loss_nats_per_token=math.log(4),
            token_count=5,
            raw_byte_count=10,
        )

        self.assertAlmostEqual(bpb, 1.0, places=6)

    def test_efficiency_script_reports_artifact_size_and_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "model.bin"
            artifact.write_bytes(b"abcd")
            output = Path(tmpdir) / "score.json"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/efficiency_score.py",
                    f"--artifact={artifact}",
                    "--loss_nats=1.3862943611198906",
                    "--tokens=5",
                    "--raw_bytes=10",
                    "--max_artifact_mib=1",
                    f"--output_json={output}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            score = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(score["artifact"]["bytes"], 4)
        self.assertTrue(score["artifact"]["within_limit"])
        self.assertAlmostEqual(score["metrics"]["bits_per_byte"], 1.0, places=6)


class ResearchRoadmapDocTests(unittest.TestCase):
    def test_public_research_roadmap_covers_safe_private_extractions(self):
        roadmap = (ROOT / "docs" / "optimizer-auxiliary-roadmap.md").read_text(encoding="utf-8")

        for phrase in [
            "Sekant/SCTR",
            "optimizer stability",
            "multi-state",
            "future-token",
            "update-budget accounting",
            "Do not copy private code wholesale",
        ]:
            self.assertIn(phrase, roadmap)

    def test_tokenizer_plan_defines_multilingual_acceptance_tests(self):
        plan = (ROOT / "docs" / "tokenizer-plan.md").read_text(encoding="utf-8")

        for phrase in [
            "BPE",
            "SentencePiece",
            "Italian",
            "German",
            "Polish",
            "diacritics",
            "round-trip",
            "bits per byte",
            "No heavy tokenizer dependency",
        ]:
            self.assertIn(phrase, plan)


if __name__ == "__main__":
    unittest.main()
