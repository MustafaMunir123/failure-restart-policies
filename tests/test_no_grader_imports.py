import ast
from pathlib import Path

FORBIDDEN = ("run-test", "run_test", "assertion", "golden", "expected_state",
             "grader", "test_result")

CONTROLLER_DIR = Path(__file__).parent.parent / "controller"


def test_no_grader_imports_or_references():
    """CI lint: controller must never touch grader/check modules or hidden answers."""
    offenders = []
    for py in CONTROLLER_DIR.glob("*.py"):
        src = py.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if any(f in a.name.lower() for f in FORBIDDEN):
                        offenders.append(f"{py.name}: import {a.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if any(f in node.module.lower() for f in FORBIDDEN):
                    offenders.append(f"{py.name}: from {node.module}")
        # crude textual check on non-docstring content
        if "expected_state" in src.replace("Never expected state", ""):
            offenders.append(f"{py.name}: mentions expected_state")
    assert not offenders, offenders
