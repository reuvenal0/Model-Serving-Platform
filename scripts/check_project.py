"""Portable syntax/import/layer verification, without opening application resources."""

import ast
import importlib
from pathlib import Path
import pkgutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def layer_errors(root: Path) -> list[str]:
    errors = []
    for layer in ("domain", "interfaces", "services"):
        banned = ("fastapi", "sqlalchemy", "app.infrastructure", "app.api")
        if layer == "domain":
            banned += ("pydantic",)
        for path in (root / "app" / layer).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = (
                    [item.name for item in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for name in names
                    for prefix in banned
                ):
                    errors.append(
                        f"{path.relative_to(root)}:{node.lineno}: invalid layer dependency"
                    )
    return errors


def main() -> None:
    sys.path.insert(0, str(ROOT))
    for folder in ("app", "examples", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    errors = layer_errors(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    import app

    for module in pkgutil.walk_packages(app.__path__, app.__name__ + "."):
        importlib.import_module(module.name)
    print("Syntax, application imports, and layer dependencies: PASS")


if __name__ == "__main__":
    main()
