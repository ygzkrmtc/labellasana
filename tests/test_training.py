"""Eğitim altyapısı ve model kütüphanesi testleri (Qt gerektirmez)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import training  # noqa: E402


class TestPaths(unittest.TestCase):
    def test_venv_python_matches_platform(self):
        path = training.venv_python(os.path.join("proje", ".venv-train"))
        if os.name == "nt":
            self.assertTrue(path.endswith("python.exe"))
            self.assertIn("Scripts", path)
        else:
            self.assertTrue(path.endswith("python"))
            self.assertIn("bin", path)

    def test_find_training_python_returns_none_when_absent(self):
        self.assertIsNone(training.find_training_python(""))
        self.assertIsNone(training.find_training_python("/olmayan/klasor"))

    def test_directories_are_under_project(self):
        project = os.path.join("a", "b")
        for func in (training.work_dir, training.dataset_dir, training.models_dir,
                     training.training_env_dir):
            self.assertTrue(func(project).startswith(project))


class TestDefaults(unittest.TestCase):
    def test_more_epochs_for_small_datasets(self):
        small = training.default_options(30)
        large = training.default_options(5000)
        self.assertGreater(small["epochs"], large["epochs"])

    def test_cache_only_for_small_datasets(self):
        self.assertTrue(training.default_options(200)["cache"])
        self.assertFalse(training.default_options(9000)["cache"])

    def test_batch_is_auto_by_default(self):
        self.assertEqual(training.default_options(100)["batch"], 0)

    def test_zero_images_is_safe(self):
        options = training.default_options(0)
        self.assertGreater(options["epochs"], 0)


class TestJob(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_config_contains_absolute_paths(self):
        config = training.build_config("data.yaml", self.dir,
                                       training.default_options(100))
        self.assertTrue(os.path.isabs(config["data"]))
        self.assertTrue(os.path.isabs(config["project"]))
        self.assertEqual(config["name"], training.RUN_NAME)

    def test_write_job_creates_valid_python(self):
        config = training.build_config("data.yaml", self.dir,
                                       training.default_options(100))
        script_path, config_path = training.write_job(self.dir, config)
        self.assertTrue(os.path.isfile(script_path))
        self.assertTrue(os.path.isfile(config_path))

        with open(script_path, encoding="utf-8") as handle:
            source = handle.read()
        compile(source, script_path, "exec")            # sözdizimi geçerli mi
        self.assertIn(training.MARK_ONNX, source)
        self.assertIn(training.MARK_BEST, source)

        with open(config_path, encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual(written["epochs"], config["epochs"])

    def test_setup_commands_order(self):
        commands = training.setup_commands("python", os.path.join(self.dir, "env"))
        self.assertEqual(len(commands), 3)
        self.assertIn("venv", commands[0][1])
        self.assertIn("ultralytics", commands[2][1])


class TestParsing(unittest.TestCase):
    def test_epoch_line(self):
        line = ("       12/100      1.24G      1.234      3.456      1.234"
                "         27        640: 100%|#####| 5/5")
        parsed = training.parse_line(line)
        self.assertEqual(parsed["kind"], "epoch")
        self.assertEqual(parsed["current"], 12)
        self.assertEqual(parsed["total"], 100)

    def test_metrics_line(self):
        line = "                   all         12         48      0.812      0.708      0.751     0.4891"
        parsed = training.parse_line(line)
        self.assertEqual(parsed["kind"], "metrics")
        self.assertAlmostEqual(parsed["map50"], 0.751, places=3)
        self.assertAlmostEqual(parsed["map50_95"], 0.4891, places=4)
        self.assertAlmostEqual(parsed["precision"], 0.812, places=3)

    def test_markers(self):
        best = training.parse_line(training.MARK_BEST + "C:/x/best.pt")
        self.assertEqual(best, {"kind": "best", "path": "C:/x/best.pt"})
        onnx = training.parse_line(training.MARK_ONNX + "C:/x/best.onnx")
        self.assertEqual(onnx["path"], "C:/x/best.onnx")
        error = training.parse_line(training.MARK_ERROR + "patladi")
        self.assertEqual(error["message"], "patladi")
        device = training.parse_line(training.MARK_DEVICE + "cuda")
        self.assertEqual(device["device"], "cuda")

    def test_ordinary_lines_ignored(self):
        for line in ("", "Ultralytics 8.1.0 basliyor", "Scanning labels...",
                     "0/0 hatali"):
            self.assertIsNone(training.parse_line(line))

    def test_epoch_zero_is_not_progress(self):
        self.assertIsNone(training.parse_line("   0/100  ..."))

    def test_probe_parsing(self):
        payload = {"python": "3.12.4", "ultralytics": "8.1.0", "cuda": True,
                   "gpu": "RTX 4060", "torch": "2.3.0"}
        output = "gurultu\nPROBE::" + json.dumps(payload) + "\nbaska satir"
        self.assertEqual(training.parse_probe(output), payload)

    def test_probe_missing_returns_empty(self):
        self.assertEqual(training.parse_probe("hicbir sey"), {})

    def test_probe_command_is_runnable(self):
        """Sonda betiği gerçekten çalışmalı (bu Python'da denenir)."""
        import subprocess
        command = training.probe_command(sys.executable)
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        info = training.parse_probe(result.stdout)
        self.assertIn("python", info)


class TestModelLibrary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.library = training.ModelLibrary(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fake_onnx(self, name="best.onnx"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(b"onnx")
        return path

    def test_empty_library(self):
        self.assertEqual(self.library.list_models(), [])

    def test_register_and_list(self):
        source = self._fake_onnx()
        target = self.library.register(source, "aku_v1")
        self.assertTrue(os.path.isfile(target))
        self.assertTrue(target.endswith("aku_v1.onnx"))
        self.assertEqual(len(self.library.list_models()), 1)

    def test_register_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.library.register(os.path.join(self.dir, "yok.onnx"), "x")

    def test_unsafe_name_is_cleaned(self):
        target = self.library.register(self._fake_onnx(), "a/b:c *?d")
        self.assertNotIn("/", os.path.basename(target))
        self.assertNotIn(":", os.path.basename(target))

    def test_metadata_round_trip(self):
        target = self.library.register(self._fake_onnx(), "aku_v1")
        self.library.write_metadata(target, {"classes": ["aku"], "map50": 0.83})
        metadata = self.library.read_metadata(target)
        self.assertEqual(metadata["classes"], ["aku"])
        self.assertAlmostEqual(metadata["map50"], 0.83)

    def test_metadata_missing_is_empty(self):
        target = self.library.register(self._fake_onnx(), "aku_v1")
        self.assertEqual(self.library.read_metadata(target), {})

    def test_newest_model_is_first(self):
        import time
        first = self.library.register(self._fake_onnx("a.onnx"), "eski")
        time.sleep(0.02)
        second = self.library.register(self._fake_onnx("b.onnx"), "yeni")
        models = self.library.list_models()
        self.assertEqual(os.path.basename(models[0]), os.path.basename(second))
        self.assertEqual(len(models), 2)
        self.assertTrue(os.path.isfile(first))


class TestNaming(unittest.TestCase):
    def test_single_class_versioning(self):
        self.assertEqual(training.suggest_model_name(["aku"], []), "aku_v1")
        self.assertEqual(
            training.suggest_model_name(["aku"], ["/x/aku_v1.onnx"]), "aku_v2")

    def test_multi_class_uses_generic_stem(self):
        self.assertEqual(training.suggest_model_name(["aku", "msda"], []), "model_v1")

    def test_no_classes(self):
        self.assertTrue(training.suggest_model_name([], []).startswith("model"))

    def test_safe_name(self):
        self.assertEqual(training.safe_name(" a b "), "a_b")
        self.assertEqual(training.safe_name("../kotu"), "kotu")
        self.assertEqual(training.safe_name(""), "")

    def test_describe_model(self):
        text = training.describe_model("/x/aku_v1.onnx",
                                       {"classes": ["aku"], "map50": 0.812})
        self.assertIn("aku_v1", text)
        self.assertIn("mAP50", text)
        self.assertIn("1 sınıf", text)

    def test_describe_model_without_metadata(self):
        self.assertEqual(training.describe_model("/x/model.onnx", {}), "model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
