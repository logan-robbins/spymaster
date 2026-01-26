import argparse
import os
import shutil
from pathlib import Path

DEFAULT_ROOT = "/Users/loganrobbins/Documents/Unreal Projects/MarketWindTunnel"

JUNK_DIRS = {
    "Binaries",
    "Intermediate",
    "Saved",
    "DerivedDataCache",
    ".vs",
    ".idea",
    ".vscode",
}

JUNK_DIR_SUFFIXES = {".xcodeproj", ".xcworkspace"}

JUNK_FILE_SUFFIXES = {
    ".sln",
    ".suo",
    ".opensdf",
    ".sdf",
    ".VC.db",
    ".VC.opendb",
    ".code-workspace",
}

JUNK_FILE_NAMES = {".DS_Store"}


def validate_unreal_root(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root path not found: {root}")
    if not list(root.glob("*.uproject")):
        raise SystemExit(f"No .uproject found in {root} (refusing to clean)")


def remove_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] rm -rf {path}")
        return
    shutil.rmtree(path)
    print(f"deleted dir: {path}")


def remove_file(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] rm {path}")
        return
    path.unlink()
    print(f"deleted file: {path}")


def clean_project(root: Path, dry_run: bool) -> None:
    removed_dirs = 0
    removed_files = 0

    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        current_path = Path(current_root)

        to_remove = []
        for dirname in list(dirnames):
            if dirname in JUNK_DIRS or any(dirname.endswith(s) for s in JUNK_DIR_SUFFIXES):
                to_remove.append(dirname)
                dirnames.remove(dirname)

        for dirname in to_remove:
            remove_dir(current_path / dirname, dry_run)
            removed_dirs += 1

        for filename in filenames:
            if filename in JUNK_FILE_NAMES or any(
                filename.endswith(s) for s in JUNK_FILE_SUFFIXES
            ):
                remove_file(current_path / filename, dry_run)
                removed_files += 1

    print(f"removed_dirs={removed_dirs} removed_files={removed_files}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete Unreal project build/cache artifacts."
    )
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    validate_unreal_root(root)
    clean_project(root, args.dry_run)


if __name__ == "__main__":
    main()
