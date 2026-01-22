import re
import subprocess
import sys
from pathlib import Path

# [1, 2, 5, 10, 20, 50, 100, 150, 250, 500, 1000, 2500]
MIN_WORD_OCCURENCE_VALUES = [2500]


def update_min_word_occurrence(config_path: Path, value: int) -> None:
    config_text = config_path.read_text()
    pattern = r"^(MIN_WORD_OCCURENCE\s*=\s*).*$"
    updated_text, count = re.subn(pattern, rf"\g<1>{value}", config_text, flags=re.M)
    if count != 1:
        raise ValueError("MIN_WORD_OCCURENCE line not found in config.py")
    config_path.write_text(updated_text)


def run_script(root: Path, script_name: str) -> None:
    subprocess.run([sys.executable, script_name], check=True, cwd=root)


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "config.py"

    for value in MIN_WORD_OCCURENCE_VALUES:
        print(f"=== MIN_WORD_OCCURENCE={value} ===")
        update_min_word_occurrence(config_path, value)
        run_script(root, "tf-idf_embeddings.py")
        run_script(root, "review_prediction_model_using_regression.py")

    update_min_word_occurrence(config_path, 250)


if __name__ == "__main__":
    main()
