"""
Breast Cancer Detection - Launch Streamlit App in Google Colab
This script launches the Streamlit app and creates a public URL using ngrok
"""

import os
import subprocess
import time
from pyngrok import ngrok

print("=" * 80)
print("LAUNCHING STREAMLIT APP")
print("=" * 80)

# Step 1: Set up ngrok authentication
print("\n[STEP 1/4] Setting up ngrok...")
print("\n⚠️  IMPORTANT: You need an ngrok auth token!")
print("   1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken")
print("   2. Sign up/login (it's free)")
print("   3. Copy your auth token")
print("   4. Paste it below when prompted\n")

ngrok_token = input("Enter your ngrok auth token: ").strip()

if not ngrok_token:
    print("❌ No token provided. Cannot proceed.")
    print("   Please get your free token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    exit()

try:
    ngrok.set_auth_token(ngrok_token)
    print("✅ ngrok authentication successful!")
except Exception as e:
    print(f"❌ ngrok authentication failed: {e}")
    exit()

# Step 2: Create streamlit config
print("\n[STEP 2/4] Configuring Streamlit...")

config_dir = os.path.expanduser("~/.streamlit")
os.makedirs(config_dir, exist_ok=True)

config_content = """
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""

with open(f"{config_dir}/config.toml", "w") as f:
    f.write(config_content)

print("✅ Streamlit configured for Colab")

# Step 3: Start Streamlit in background
print("\n[STEP 3/4] Starting Streamlit server...")

# Kill any existing streamlit processes
os.system("pkill -f streamlit")
time.sleep(2)

# Start streamlit in background
process = subprocess.Popen(
    ["streamlit", "run", "streamlit_app.py", "--server.port=8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("   Waiting for server to start...")
time.sleep(5)

print("✅ Streamlit server started")

# Step 4: Create public URL with ngrok
print("\n[STEP 4/4] Creating public URL with ngrok...")

try:
    # Open ngrok tunnel
    public_url = ngrok.connect(8501)
    print("\n" + "=" * 80)
    print("✅ APP LAUNCHED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n📋 INSTRUCTIONS:")
    print("   1. Click the URL above (or copy-paste into browser)")
    print("   2. The Streamlit dashboard will open in a new tab")
    print("   3. Upload mammogram images to test predictions")
    print("   4. Keep this cell running - if you stop it, the URL will stop working")
    print("\n⚠️  IMPORTANT:")
    print("   - The URL is public but temporary (valid while this cell runs)")
    print("   - To stop the app: Interrupt this cell or restart the runtime")
    print("   - To restart: Run this cell again (you may get a new URL)")
    print("\n" + "=" * 80)
    
    # Keep the process running
    print("\n⏳ App is running... (Press Ctrl+C to stop)")
    process.wait()
    
except KeyboardInterrupt:
    print("\n\n🛑 Stopping app...")
    process.terminate()
    ngrok.kill()
    print("✅ App stopped successfully")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    process.terminate()
    ngrok.kill()