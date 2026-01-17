import os
import subprocess
import sys

def install_otx():
    print("Installing OpenVINO Training Extensions (OTX)...")
    try:
        # Installing 'otx' package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "otx"])
        print("\n✅ OTX Installation Complete!")
        print("You can now verify installation by running: otx --help")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation Failed: {e}")

if __name__ == "__main__":
    print("=== OpenVINO Fine-Tuning Tools Installer ===")
    install_otx()
