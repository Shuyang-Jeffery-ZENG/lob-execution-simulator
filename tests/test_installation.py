"""Require a wheel installed in the active venv, independent of the checkout."""
import json
from pathlib import Path
import subprocess
import sys


MODULES = (
    "lob_lab", "lob_lab.costs", "lob_lab.book", "lob_lab.valuation",
    "lob_lab.execution", "lob_lab.policy", "lob_lab.replay", "lob_lab.timed_execution",
    "lob_sim", "lob_sim.models", "lob_sim.recovery", "lob_sim.engine", "lob_sim.demo",
)


def isolated_python(cwd, script, *args):
    return subprocess.run(
        [sys.executable, "-I", "-c", script, *args],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


def test_all_modules_belong_to_the_installed_wheel_in_current_venv(tmp_path):
    script = """
import importlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
import sysconfig

modules = json.loads(sys.argv[1])
dist = distribution("lob-execution-simulator")
direct_url = dist.read_text("direct_url.json")
paths = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
owned = {str(file): str(Path(dist.locate_file(file)).resolve()) for file in dist.files or ()}
print(json.dumps({
    "version": dist.version,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "site_dirs": [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")],
    "modules": paths,
    "owned_files": owned,
    "wheel_metadata": dist.read_text("WHEEL"),
    "direct_url": json.loads(direct_url) if direct_url else None,
    "entry_points": {entry.name: entry.value for entry in dist.entry_points if entry.group == "console_scripts"},
}))
"""
    run = isolated_python(tmp_path, script, json.dumps(MODULES))
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    assert data["version"] == "0.1.0"
    assert Path(data["prefix"]).resolve() == Path(sys.prefix).resolve()
    assert data["prefix"] != data["base_prefix"], "Run the checks in a clean venv with the wheel installed."
    sites = tuple(Path(value).resolve() for value in data["site_dirs"])
    assert all(site.is_relative_to(Path(sys.prefix).resolve()) for site in sites)
    assert data["wheel_metadata"] and "Wheel-Version:" in data["wheel_metadata"]
    direct_url = data["direct_url"]
    assert not direct_url or not direct_url.get("dir_info", {}).get("editable", False)
    assert data["entry_points"]["lob-execution-demo"] == "lob_sim.demo:main"
    assert set(data["modules"]) == set(MODULES)
    for name, filename in data["modules"].items():
        path = Path(filename)
        assert any(path.is_relative_to(site) for site in sites), f"Module is outside installed site-packages: {name}"
        relative = name.replace(".", "/") + ("/__init__.py" if name in {"lob_lab", "lob_sim"} else ".py")
        assert data["owned_files"][relative] == str(path), f"Module is not owned by this wheel: {name}"


def test_registered_console_entry_point_runs_isolated_from_checkout(tmp_path):
    script = """
from importlib.metadata import distribution
entry = next(entry for entry in distribution("lob-execution-simulator").entry_points
             if entry.group == "console_scripts" and entry.name == "lob-execution-demo")
raise SystemExit(entry.load()())
"""
    run = isolated_python(tmp_path, script, "--case", "gap-recovery")
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert [row["case"] for row in report["cases"]] == ["gap-recovery"]
    assert report["cases"][0]["result"]["actual_state"]["notional"] == "404"
    assert list(tmp_path.iterdir()) == []
