from pathlib import Path
from datetime import datetime

def create_history_file(identity: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    path = Path("data/history")
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{identity}_history_{ts}.txt"
    return file

def log_action(file: Path, action: str, desc: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with open(file, "a") as f:
        f.write(f"[{ts}] {action} - {desc}\n")