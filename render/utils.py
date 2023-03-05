import sys
import subprocess


def install_requirements():
    """Install required packages from `requirements.txt`. Used when running scripts with the Python version packaged 
    with Blender."""
    # Upgrade pip
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Load list of requirements from `requirements.txt`
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    # Install required packages
    for r in required:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', r])

    # subprocess.check_call([sys.executable, 'setup.py', 'develop'])