"""
Quick script to check if all required packages are installed
"""

import sys

required_packages = {
    'torch': 'torch',
    'transformers': 'transformers',
    'numpy': 'numpy',
    'tqdm': 'tqdm',
}

missing_packages = []

print("Checking required packages...")
print("-" * 50)

for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✓ {package_name:<20} - Installed")
    except ImportError:
        print(f"✗ {package_name:<20} - NOT FOUND")
        missing_packages.append(package_name)

print("-" * 50)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    print("\nTo install missing packages, run:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n✓ All required packages are installed!")
    print("\nYou can now run the evaluation script:")
    print("  python evaluate_models_v2.py")
    sys.exit(0)
