"""Every pack is loadable, internally consistent, and free of site-specific detail."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import support  # noqa: F401  (pack and registry setup)
import port
from cluster import identity

HARNESS = support.HARNESS
PACKS = sorted(path for path in (support.ROOT / "ports").iterdir() if (path / "port.json").is_file())
REAL = [path for path in PACKS if path.name != "template"]


def pinned_corpus_hash(pack):
    match = re.search(r'^CORPUS_SHA256 = ["\']([0-9a-f]{64})["\']', (pack / "evaluate.py").read_text(), re.M)
    return match and match.group(1)


class PortPackTests(unittest.TestCase):
    def test_every_pack_loads_and_freezes(self):
        self.assertEqual({path.name for path in REAL}, {"chronos2", "esm2", "nllb", "parakeet"})
        for pack in PACKS:
            with self.subTest(pack=pack.name), tempfile.TemporaryDirectory() as directory:
                loaded = port.load_port(pack)
                self.assertIn(loaded["key"], loaded["delivery_models"])
                self.assertIn("CONTRACT.md", loaded["docs"])
                self.assertIn(loaded["title"], loaded["objective"] + loaded["title"])
                port.freeze(loaded, directory)
                self.assertEqual(port.load_port(directory)["key"], loaded["key"])

    def test_corpus_identity_agrees_between_manifest_evaluator_and_file(self):
        for pack in REAL:
            with self.subTest(pack=pack.name):
                loaded = port.load_port(pack)
                self.assertEqual(loaded["corpus_sha256"], pinned_corpus_hash(pack))
                if loaded["corpus_path"]:
                    self.assertEqual(identity(json.loads(loaded["corpus_path"].read_text())), loaded["corpus_sha256"])

    def test_shipped_corpora_build_valid_suites_for_every_stage(self):
        # Each pack's evaluator is a module named "evaluate", so build suites in a
        # fresh interpreter per pack.
        script = (
            "import json, sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]\n"
            "import models; models.use_registry(sys.argv[1] + '/models.json')\n"
            "import evaluate, portforge_eval\n"
            "corpus = json.load(open(sys.argv[3]))\n"
            "for stage in ('smoke', 'bringup', 'full'):\n"
            "    spec, _ = evaluate.make_suite(corpus, stage)\n"
            "    portforge_eval.check_suite(spec)\n"
            "    print(stage, len(spec['cases']))\n")
        for pack in REAL:
            loaded = port.load_port(pack)
            if not loaded["corpus_path"]:
                continue
            with self.subTest(pack=pack.name):
                done = subprocess.run([sys.executable, "-c", script, str(pack), str(HARNESS), str(loaded["corpus_path"])],
                                      capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual([line.split()[0] for line in done.stdout.split("\n") if line], ["smoke", "bringup", "full"])
    def test_check_suite_rejects_consistency_cases_without_a_target(self):
        from portforge_eval import check_suite
        spec = {"stage": "full", "cases": [{"name": "batch4", "clips": ["a", "b"]},
                                           {"name": "revisit", "clips": ["a", "b"], "compare_to": "batch",
                                            "compare_rows": [0, 1]}]}
        with self.assertRaisesRegex(ValueError, "missing case batch"):
            check_suite(spec)
        spec["cases"][1]["compare_to"] = "batch4"
        self.assertIs(check_suite(spec), spec)

    def test_every_evaluator_upload_ships_the_shared_module(self):
        # evaluate.py imports portforge_eval; any site that uploads one must upload both.
        import ast
        source = (HARNESS / "portforge_loop.py").read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Tuple) and all(isinstance(e, ast.Constant) for e in node.elts):
                names = {e.value for e in node.elts}
                if "evaluate.py" in names and "models.py" in names:
                    self.assertIn("portforge_eval.py", names, ast.get_source_segment(source, node))
        self.assertIn("'/portforge_eval.py'", (HARNESS / "runner.py").read_text())

    def test_agent_documents_use_the_shared_tool_names(self):
        stale = re.compile(r"\b(?:nllb|chronos|esm2?|parakeet)_(files|read|write|run|evaluate|status|log|submit)\b")
        for path in list((support.ROOT / "ports").glob("*/*.md")) + list((HARNESS / "guides").glob("*.md")):
            with self.subTest(path=path.name):
                self.assertIsNone(stale.search(path.read_text()), path)

    def test_tree_has_no_site_specific_hosts_users_or_paths(self):
        pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|/home/(?!<)[a-z]|/Users/[A-Za-z]|abror|murodbe|id_tt_box|"
                             r"google_compute|tt-blackhole-\d")
        # RFC 5737 documentation ranges are the intended placeholders in test fixtures.
        documentation = ("192.0.2.", "198.51.100.", "203.0.113.")
        excluded = {".git", ".venv", "venv", "runs", "assets", "dist", "build",
                    "__pycache__", "portforge_tt.egg-info"}
        for path in support.ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".md", ".json", ".yaml", ".sh"):
                continue
            if any(part in excluded for part in path.parts):
                continue
            if path.name == "corpus.json" or path.name == "README.md" or path == Path(__file__):
                continue
            for match in pattern.finditer(path.read_text()):
                if match.group(0) == "127.0.0.1" or match.group(0).startswith(documentation):
                    continue
                with self.subTest(path=str(path.relative_to(support.ROOT)), match=match.group(0)):
                    self.fail(f"site-specific detail {match.group(0)!r} in {path.relative_to(support.ROOT)}")


if __name__ == "__main__":
    unittest.main()
