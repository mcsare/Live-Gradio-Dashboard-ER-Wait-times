"""Create all tables, then load the static CSVs. Safe to run more than once.

  python scripts/init_db.py           uses DATABASE_URL
  python scripts/init_db.py --local   uses data/local.db
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_engine, metadata  # noqa: E402
from scripts.load_static import load_all  # noqa: E402

if __name__ == "__main__":
    local = "--local" in sys.argv
    engine = get_engine(local=local)
    metadata.create_all(engine)
    print(f"Tables ready on {engine.url.render_as_string(hide_password=True)}")
    load_all(engine)
