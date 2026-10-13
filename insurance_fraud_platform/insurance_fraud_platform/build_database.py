"""Command-line helper: python build_database.py  (rebuilds data, warehouse and risk scores)."""
from src.pipeline import build_database

if __name__ == "__main__":
    print("Building warehouse ...")
    for key, value in build_database(regenerate=True).items():
        print(f"  {key:<24} {value}")
    print("Done. Start the app with:  streamlit run app.py")
