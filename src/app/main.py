import sys
from pathlib import Path
from streamlit.web import cli as stcli

# ---------- MAIN ENTRY POINT ----------

def main():
    """
    Entry point to run the Streamlit application programmatically.
    Equivalent to running: streamlit run app.py
    """
    # 1. Resolve the path to app.py
    # This assumes app.py is in the same directory as main.py
    app_path = Path(__file__).parent.parent.parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)

    print(f"Starting NEU-DET Application...")
    print(f"   Target: {app_path}")
    
    # 2. Construct the argument list mimicking the command line
    # sys.argv[0] is the script name, so we set it to 'streamlit'
    sys.argv = ["streamlit", "run", str(app_path)]
    
    # 3. Launch Streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()