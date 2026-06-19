from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from project_paths import DATA_RAW, ensure_project_dirs, rel


UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def download_uci() -> bool:
    uci_dir = DATA_RAW / "taiwan"
    uci_dir.mkdir(parents=True, exist_ok=True)
    xls_path = uci_dir / "default of credit card clients.xls"
    if xls_path.exists():
        print(f"UCI file already exists: {rel(xls_path)}")
        return True

    errors: list[str] = []
    for url in UCI_URLS:
        try:
            suffix = ".zip" if url.endswith(".zip") else ".xls"
            tmp_path = uci_dir / f"uci_default_credit{suffix}"
            download_file(url, tmp_path)
            if suffix == ".zip":
                with zipfile.ZipFile(tmp_path) as zf:
                    zf.extractall(uci_dir)
                tmp_path.unlink(missing_ok=True)
                candidates = list(uci_dir.glob("*.xls"))
                if candidates:
                    candidates[0].rename(xls_path)
            else:
                tmp_path.rename(xls_path)
            if xls_path.exists():
                print(f"Saved UCI file: {rel(xls_path)}")
                return True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")

    print("UCI download failed.")
    for error in errors:
        print(f"- {error}")
    return False


def download_home_credit() -> bool:
    home_dir = DATA_RAW / "home_credit"
    home_dir.mkdir(parents=True, exist_ok=True)
    kaggle_config = DATA_RAW / "kaggle_config"
    kaggle_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_config))
    train_file = home_dir / "application_train.csv"
    if train_file.exists():
        print(f"Home Credit file already exists: {rel(train_file)}")
        return True

    kaggle = shutil.which("kaggle")
    local_kaggle = Path(sys.executable).with_name("kaggle.exe")
    if not kaggle and local_kaggle.exists():
        kaggle = str(local_kaggle)
    if not kaggle:
        print("Kaggle CLI is not installed; skipping Home Credit download.")
        print("Install dependencies and configure Kaggle credentials, then run:")
        print("  kaggle competitions download -c home-credit-default-risk -p data/raw/home_credit")
        print(f"Project-local Kaggle config directory: {rel(kaggle_config)}")
        return False

    cmd = [
        kaggle,
        "competitions",
        "download",
        "-c",
        "home-credit-default-risk",
        "-p",
        str(home_dir),
    ]
    print("Running Kaggle download...")
    result = subprocess.run(cmd, cwd=DATA_RAW.parents[1], text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"Project-local Kaggle config directory: {rel(kaggle_config)}")
        return False

    for zip_path in home_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(home_dir)
    return train_file.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download project datasets when access is available.")
    parser.add_argument("--uci-only", action="store_true", help="Only download the UCI Taiwan credit dataset.")
    parser.add_argument("--home-credit-only", action="store_true", help="Only attempt Kaggle Home Credit download.")
    args = parser.parse_args()

    ensure_project_dirs()
    ok = True
    if not args.home_credit_only:
        ok = download_uci() and ok
    if not args.uci_only:
        ok = download_home_credit() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
