"""
Launch Streamlit App in Google Colab
Creates public URL for accessing the dashboard
"""

import os
import subprocess
import time
from IPython.display import display, HTML

print("=" * 80)
print("LAUNCHING STREAMLIT APP IN GOOGLE COLAB")
print("=" * 80)

# Verify app file exists
if not os.path.exists("5_streamlit_app.py"):
    print("❌ Streamlit app file not found!")
    print("Make sure you've run all previous steps first.")
    exit(1)

print("\n✅ App file found")

# Kill any existing Streamlit processes
print("\n[1/3] Cleaning up existing processes...")
os.system("pkill -f streamlit > /dev/null 2>&1")
time.sleep(2)
print("✅ Cleanup complete")

# Start Streamlit server
print("\n[2/3] Starting Streamlit server...")

streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "5_streamlit_app.py",
     "--server.port=8501",
     "--server.address=0.0.0.0",
     "--server.headless=true",
     "--server.enableCORS=false",
     "--server.enableXsrfProtection=false"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

print("   Waiting for server to start...")
time.sleep(10)

if streamlit_proc.poll() is not None:
    print("❌ Failed to start Streamlit")
    exit(1)

print("✅ Streamlit running on localhost:8501")

# Display access instructions
print("\n[3/3] Setting up public access...")

display(HTML('''
<div style="padding: 20px; background: #e8f5e9; border: 3px solid #4caf50; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: #2e7d32;">✅ Streamlit is Running!</h2>
    
    <h3 style="color: #1976d2;">🌐 Access Your App:</h3>
    
    <div style="background: #fff; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4>Method 1: LocalTunnel (Recommended) 🚀</h4>
        <p>Run this command in a <strong>NEW</strong> Colab cell:</p>
        <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">!npx localtunnel --port 8501</pre>
        <p>Then:</p>
        <ol>
            <li>Copy the URL that appears (https://...loca.lt)</li>
            <li>Open it in a new browser tab</li>
            <li>Click "Continue" if you see a warning page</li>
        </ol>
    </div>
    
    <div style="background: #fff; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4>Method 2: Colab Port Forwarding 🔗</h4>
        <ol>
            <li>Click the 🔗 icon in the left sidebar</li>
            <li>Find port <strong>8501</strong></li>
            <li>Click to open in browser</li>
        </ol>
    </div>
    
    <p style="color: #d32f2f; font-weight: bold;">⚠️ IMPORTANT: Keep this cell running! Stopping it will stop the app.</p>
</div>
'''))

print("\n" + "=" * 80)
print("📋 QUICK START GUIDE")
print("=" * 80)
print("""
TO ACCESS YOUR APP:

Option A - LocalTunnel (easiest):
  1. Open a NEW Colab cell below
  2. Run: !npx localtunnel --port 8501
  3. Copy the URL (https://...loca.lt)
  4. Open in browser
  5. Click 'Continue' on warning page

Option B - Port Forwarding:
  1. Click 🔗 in left sidebar
  2. Find port 8501
  3. Click to open

⚠️  KEEP THIS CELL RUNNING
""")

print("\n💚 Status: RUNNING")
print("   App is ready at: http://localhost:8501")
print("   Press Stop button to shut down\n")

print("=" * 80)

# Keep alive
try:
    while streamlit_proc.poll() is None:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    streamlit_proc.terminate()
    streamlit_proc.wait()
    print("✅ Shutdown complete")