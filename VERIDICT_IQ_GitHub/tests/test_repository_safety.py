from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_private_data_directories_are_empty() -> None:
    for relative in ("data/raw", "data/interim", "data/processed"):
        files = [p for p in (REPO_ROOT / relative).rglob("*") if p.is_file() and p.name != ".gitkeep"]
        assert files == []
