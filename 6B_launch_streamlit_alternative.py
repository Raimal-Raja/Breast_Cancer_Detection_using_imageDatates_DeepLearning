"""
Fixed Streamlit Launcher for Google Colab
Proper error handling and debugging
"""

import os
import subprocess
import time
import sys
from IPython.display import display, HTML

print("=" * 80)
print("STREAMLIT APP LAUNCHER - DIAGNOSTIC MODE")
print("=" * 80)

# Step 1: Verify app file exists
print("\n[STEP 1] Checking for Streamlit app file...")
app_file = "5_streamlit_app.py"

if not os.path.exists(app_file):
    print(f"❌ ERROR: '{app_file}' not found!")
    print("\n📁 Files in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.py'):
            print(f"   - {f}")
    print("\n💡 SOLUTION: Make sure you've created the app file first!")
    print("   Run the cell that creates '5_streamlit_app.py'")
    sys.exit(1)

print(f"✅ Found: {app_file}")

# Step 2: Verify model files
print("\n[STEP 2] Checking for model files...")
if not os.path.exists('models'):
    print("⚠️  WARNING: 'models' directory not found!")
    print("   App will show 'Model Not Found' error")
    print("   Make sure you've trained the model first!")
else:
    print("✅ Models directory exists")
    if os.path.exists('models/best_model.h5'):
        print("✅ Found: best_model.h5")
    else:
        print("⚠️  WARNING: best_model.h5 not found")

# Step 3: Install dependencies
print("\n[STEP 3] Checking dependencies...")
try:
    import streamlit
    print(f"✅ Streamlit {streamlit.__version__} installed")
except ImportError:
    print("❌ Streamlit not installed!")
    print("   Installing now...")
    os.system("pip install -q streamlit plotly")
    print("✅ Installation complete")

# Step 4: Kill existing processes
print("\n[STEP 4] Cleaning up old processes...")
os.system("pkill -f streamlit > /dev/null 2>&1")
os.system("pkill -f 'streamlit run' > /dev/null 2>&1")
time.sleep(2)
print("✅ Cleanup complete")

# Step 5: Start Streamlit with error capture
print("\n[STEP 5] Starting Streamlit server...")
print("   Command: streamlit run 5_streamlit_app.py --server.port=8501")

# Create a log file for debugging
log_file = "streamlit_launch.log"

streamlit_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

print("   Waiting for startup (15 seconds)...")

# Monitor startup
startup_output = []
for i in range(15):
    time.sleep(1)
    # Check if process crashed
    if streamlit_proc.poll() is not None:
        print("\n❌ STREAMLIT CRASHED DURING STARTUP!")
        print("\n📋 Error output:")
        print("-" * 80)
        output, _ = streamlit_proc.communicate()
        print(output)
        print("-" * 80)
        print("\n💡 Common fixes:")
        print("   1. Make sure the app file is valid Python")
        print("   2. Check if all imports are available")
        print("   3. Verify the model file exists")
        sys.exit(1)
    
    print(f"   {'.' * (i+1)}", end='\r')

print("\n")

# Check if still running
if streamlit_proc.poll() is None:
    print("✅ Streamlit server started successfully!")
else:
    print("❌ Streamlit failed to stay running")
    sys.exit(1)

# Step 6: Verify it's accessible
print("\n[STEP 6] Verifying server...")
time.sleep(2)

try:
    import requests
    response = requests.get("http://localhost:8501", timeout=5)
    print("✅ Server is responding!")
except Exception as e:
    print(f"⚠️  Could not verify server: {e}")
    print("   (This might be okay - try accessing anyway)")

# Display access instructions
print("\n" + "=" * 80)
print("✅ STREAMLIT IS RUNNING!")
print("=" * 80)

display(HTML('''
<div style="padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white; margin: 20px 0;">
    <h1 style="margin: 0 0 20px 0;">🎉 Streamlit App is Live!</h1>
    
    <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h2 style="margin-top: 0;">🚀 Method 1: LocalTunnel (Easiest)</h2>
        <ol style="font-size: 16px; line-height: 1.8;">
            <li>Create a <strong>NEW CODE CELL</strong> below</li>
            <li>Copy and run: <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">!npx localtunnel --port 8501</code></li>
            <li>Wait for URL like: <code>https://xyz.loca.lt</code></li>
            <li>Click the URL to open your app</li>
            <li>If you see a warning page, click "Continue"</li>
        </ol>
    </div>
    
    <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h2 style="margin-top: 0;">🔗 Method 2: Colab Port Forwarding</h2>
        <ol style="font-size: 16px; line-height: 1.8;">
            <li>Look at the <strong>left sidebar</strong> of Colab</li>
            <li>Click the <strong>🔗 link icon</strong></li>
            <li>Find <strong>port 8501</strong> in the list</li>
            <li>Click it to open in new tab</li>
        </ol>
    </div>
    
    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 0; font-size: 18px;">⚠️ <strong>IMPORTANT:</strong> Keep this cell running! 
        Stopping it will stop the Streamlit app.</p>
    </div>
</div>
'''))

print("\n📊 Server Info:")
print(f"   Local URL: http://localhost:8501")
print(f"   Process ID: {streamlit_proc.pid}")
print(f"   Status: RUNNING")

print("\n💡 Quick Start:")
print("   1. Run in NEW cell: !npx localtunnel --port 8501")
print("   2. Copy the URL you get")
print("   3. Open it in your browser")
print("   4. Start analyzing mammograms!")

print("\n🛑 To stop: Click the stop button on this cell")
print("=" * 80)

# Keep process alive and monitor
print("\n💚 Monitoring server (press stop to exit)...\n")

try:
    line_count = 0
    while streamlit_proc.poll() is None:
        time.sleep(5)
        line_count += 1
        if line_count % 12 == 0:  # Every minute
            print(f"💚 Still running... ({line_count * 5}s)")
except KeyboardInterrupt:
    print("\n\n🛑 Shutdown requested...")
    streamlit_proc.terminate()
    streamlit_proc.wait()
    print("✅ Streamlit stopped cleanly")
except Exception as e:
    print(f"\n❌ Error: {e}")
    streamlit_proc.terminate()
finally:
    if streamlit_proc.poll() is None:
        streamlit_proc.terminate()
        print("\n✅ Cleanup complete")