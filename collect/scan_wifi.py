import subprocess
import pandas as pd
import os
from datetime import datetime

def scan_wifi_windows():
    """Scan WiFi networks and return SSID + RSSI"""
    result = subprocess.run(
        ["netsh", "wlan", "show", "networks", "mode=bssid"],
        capture_output=True, text=True
    )
    
    networks = []
    lines = result.stdout.split('\n')
    
    current_ssid = None
    current_bssid = None
    current_signal = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("SSID") and "BSSID" not in line:
            parts = line.split(":")
            if len(parts) > 1:
                current_ssid = parts[1].strip()
                
        elif line.startswith("BSSID"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                current_bssid = parts[1].strip()
                
        elif line.startswith("Signal"):
            parts = line.split(":")
            if len(parts) > 1:
                signal_percent = int(parts[1].strip().replace('%', ''))
                # Convert Windows % to dBm approximation
                current_signal = (signal_percent / 2) - 100
                
        if current_ssid and current_bssid and current_signal:
            networks.append({
                'ssid': current_ssid,
                'bssid': current_bssid,
                'rssi': current_signal
            })
            current_ssid = None
            current_bssid = None
            current_signal = None
    
    return networks

def collect_sample(location_name, samples=5):
    """Collect multiple scans at a location and average them"""
    print(f"\nCollecting data for: {location_name}")
    print("Scanning WiFi... (takes about 10 seconds)")
    
    all_networks = {}
    
    for i in range(samples):
        print(f"  Scan {i+1}/{samples}...")
        networks = scan_wifi_windows()
        for n in networks:
            key = n['bssid']
            if key not in all_networks:
                all_networks[key] = {
                    'ssid': n['ssid'],
                    'bssid': key,
                    'rssi_values': []
                }
            all_networks[key]['rssi_values'].append(n['rssi'])
    
    # Average the RSSI values
    row = {'location': location_name, 'timestamp': datetime.now()}
    for key, data in all_networks.items():
        col_name = f"rssi_{data['ssid']}_{key[-5:]}"
        row[col_name] = sum(data['rssi_values']) / len(data['rssi_values'])
    
    return row

def save_to_csv(row, filepath='data/wifi_fingerprints.csv'):
    """Append the row to CSV"""
    df_new = pd.DataFrame([row])
    
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filepath, index=False)
    else:
        df_new.to_csv(filepath, index=False)
    
    print(f"Saved! Total rows in dataset: {pd.read_csv(filepath).shape[0]}")

if __name__ == "__main__":
    print("=== PathFi Data Collector ===")
    print("Enter the location name where you are standing right now.")
    print("Examples: bedroom, living_room, kitchen, lab, corridor, canteen")
    print()
    
    location = input("Location name: ").strip().lower().replace(" ", "_")
    
    if not location:
        print("Location name cannot be empty!")
    else:
        row = collect_sample(location, samples=5)
        save_to_csv(row)
        print(f"\nDone! Data saved for '{location}'")
        print("Move to the next location and run the script again.")