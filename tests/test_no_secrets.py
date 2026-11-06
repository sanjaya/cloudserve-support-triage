"""Tests for the repository key scan (NFR-04, B-13). The fake keys are built at run time, so
this file itself never contains anything that looks like a real key."""
import importlib.util

from src import config

spec = importlib.util.spec_from_file_location("check_no_secrets", config.ROOT / "scripts" / "check_no_secrets.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def test_real_key_shapes_are_found_and_masked():
    fakes = {"OpenRouter key": "sk-or-v1-" + "a1" * 32, "Groq key": "gsk_" + "Ab3" * 15,
             "GitHub token": "ghp_" + "x9" * 20, "AWS access key": "AKIA" + "ABCD2345" * 2,
             "Hugging Face token": "hf_" + "Qz" * 18}
    for kind, key in fakes.items():
        found = scan.scan_text(f"value = '{key}'")
        assert any(k == kind for k, _ in found), kind
        assert all(key not in shown for _, shown in found)          # never printed in full


def test_key_assigned_in_config_is_found():
    assert scan.scan_text("OPENROUTER_API_KEY=" + "q7Lm" * 8)


def test_placeholders_and_normal_text_are_fine():
    for line in ["OPENROUTER_API_KEY=your_key_here", "OPENROUTER_API_KEY=", "api_key=test",
                 "self._api_key = api_key or config.OPENROUTER_API_KEY",
                 "Use key sk_live_abcdef1234567890.", "CONFIDENCE_THRESHOLD=0.80"]:
        assert scan.scan_text(line) == [], line


def test_the_repository_itself_has_no_keys():
    assert scan.scan_tracked() == []
