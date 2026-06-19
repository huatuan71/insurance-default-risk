from __future__ import annotations

import sys

from features import run_stage3_feature_engineering
from project_paths import rel


def main() -> int:
    try:
        outputs = run_stage3_feature_engineering()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 3 feature engineering failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 3 feature engineering complete.")
    for key in [
        "full",
        "train",
        "valid",
        "test",
        "catalog",
        "derived",
        "selection",
        "encoding",
        "iv",
        "notes",
        "status",
    ]:
        print(f"- {key}: {rel(outputs[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
