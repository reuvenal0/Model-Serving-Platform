"""Deliberately regenerate hashed locks for the single supported Linux target."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def wheel_entry(path: Path) -> tuple[str, str, str]:
    with zipfile.ZipFile(path) as wheel:
        metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
    name = metadata["Name"].lower().replace("_", "-")
    return name, metadata["Version"], hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "groups",
        nargs="*",
        choices=["runtime", "dev", "build"],
        default=["runtime", "dev", "build"],
    )
    args = parser.parse_args()
    for group in args.groups:
        with TemporaryDirectory(prefix="msp-lock-") as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--only-binary=:all:",
                    "--platform",
                    "manylinux_2_28_x86_64",
                    "--platform",
                    "manylinux2014_x86_64",
                    "--python-version",
                    "3.12",
                    "--implementation",
                    "cp",
                    "--abi",
                    "cp312",
                    "--dest",
                    directory,
                    "-r",
                    str(ROOT / "requirements" / f"{group}.in"),
                ],
                check=True,
            )
            entries = sorted(
                wheel_entry(path) for path in Path(directory).glob("*.whl")
            )
            header = (
                "# Locked for CPython 3.12, Linux x86_64; wheels only.\n"
                "# Exact wheel hashes; regenerate deliberately with scripts/lock_dependencies.py.\n"
            )
            lines = [
                f"{name}=={version} --hash=sha256:{digest}"
                for name, version, digest in entries
            ]
            (ROOT / "requirements" / f"{group}.lock").write_text(
                header + "\n".join(lines) + "\n", encoding="utf-8"
            )
            print(f"{group}: {len(entries)} pinned distributions")


if __name__ == "__main__":
    main()
