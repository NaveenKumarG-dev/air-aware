#!/usr/bin/env python3
"""
Air Quality Forecasting System - Main Runner
Run all phases in sequence: Exploration -> Preprocessing -> Features -> Modeling
"""

import subprocess
import sys


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)

    try:
        result = subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
        print(f"\n✅ {description} - COMPLETED\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED\n")
        return False


def main():
    print("\n" + "=" * 70)
    print("AIR QUALITY FORECASTING SYSTEM - FULL PIPELINE")
    print("=" * 70 + "\n")

    scripts = [
        ("01_exploration.py", "Phase 1: Data Exploration"),
        ("02_preprocessing.py", "Phase 2: Data Preprocessing"),
        ("03_features.py", "Phase 3: Feature Engineering"),
        ("04_modeling.py", "Phase 4: Model Training"),
    ]

    results = {}
    for script, description in scripts:
        results[description] = run_script(script, description)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    for desc, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {desc}")

    if all(results.values()):
        print("\n" + "=" * 70)
        print("ALL PHASES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nTo launch the dashboard:")
        print("  uv run streamlit run app.py")
    else:
        print("\n⚠️ Some phases failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
