"""Launch the Streamlit web interface."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the path to the web app
    app_path = Path(__file__).parent / "src" / "synthetic_data_gen" / "interface" / "web_app.py"
    
    print("=" * 60)
    print("Starting Streamlit Web Interface...")
    print("=" * 60)
    print()
    print("The interface will open in your browser automatically.")
    print("If it doesn't, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)

