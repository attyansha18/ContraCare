import subprocess
import sys
import os
from pathlib import Path
import venv

def create_virtual_environment():
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        try:
            # Create virtual environment
            venv.create(venv_path, with_pip=True)
            
            # On Windows, we need to ensure pip is installed
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python.exe"
                subprocess.run([str(python_path), "-m", "ensurepip", "--upgrade"], check=True)
                subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            print("Virtual environment created successfully!")
            return True
        except Exception as e:
            print(f"Error creating virtual environment: {e}")
            return False
    else:
        print("Virtual environment already exists.")
        return True

def install_requirements():
    print("Installing requirements...")
    try:
        if sys.platform == "win32":
            pip_path = Path(".venv/Scripts/pip.exe")
            python_path = Path(".venv/Scripts/python.exe")
        else:
            pip_path = Path(".venv/bin/pip")
            python_path = Path(".venv/bin/python")
        
        if not pip_path.exists():
            # If pip doesn't exist, try installing it
            subprocess.run([str(python_path), "-m", "ensurepip", "--upgrade"], check=True)
            subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"Error installing requirements: {e}")
        return False

def download_data():
    print("Downloading and preparing data...")
    try:
        if sys.platform == "win32":
            python_path = Path(".venv/Scripts/python.exe")
        else:
            python_path = Path(".venv/bin/python")
        
        subprocess.run([str(python_path), "download_data.py"], check=True)
        print("Data downloaded and prepared successfully!")
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def train_model():
    print("Training model...")
    try:
        if sys.platform == "win32":
            python_path = Path(".venv/Scripts/python.exe")
        else:
            python_path = Path(".venv/bin/python")
        
        subprocess.run([str(python_path), "train_model.py"], check=True)
        print("Model trained successfully!")
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def main():
    print("Setting up ContraCare...")
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment. Please try again.")
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please try again.")
        return
    
    # Download and prepare data
    if not download_data():
        print("Failed to download data. Please try again.")
        return
    
    # Train model
    if not train_model():
        print("Failed to train model. Please try again.")
        return
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    if sys.platform == "win32":
        print("1. Activate virtual environment: .venv\\Scripts\\activate")
    else:
        print("1. Activate virtual environment: source .venv/bin/activate")
    print("2. Run the application: streamlit run app.py")

if __name__ == "__main__":
    main() 