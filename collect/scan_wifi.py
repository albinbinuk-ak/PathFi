import subprocess
import pandas as pd
import os
from datetime import datetime

# ── KNOWN MOBILE DEVICE MAC PREFIXES (OUI) ────────────────────
MOBILE_OUI_PREFIXES = {
    # Apple (iPhone hotspots)
    'a4:c3:f0', 'f0:db:e2', 'bc:92:6b', 'fc:e9:98', 'dc:2b:2a',
    'a8:be:27', 'f4:f1:5a', '8c:85:90', 'bc:d0:74', 'f0:b4:79',
    'ac:bc:32', '70:3c:69', 'e8:8d:28', 'dc:a9:04', 'cc:08:8d',
    'f8:27:93', 'a4:d1:8c', '60:f8:1d', '18:65:90', 'bc:52:b7',

    # Samsung
    '8c:77:12', 'cc:07:ab', 'f4:42:8f', '50:a4:c8', 'e4:92:fb',
    '8c:c8:cd', 'b0:72:bf', '50:01:bb', 'fc:a1:3e', 'a0:0b:ba',
    '14:49:e0', '34:23:ba', 'a4:eb:d3', '2c:ae:2b', '8c:f5:a3',
    'c0:bd:d1', '20:13:e0', '34:14:5f', '90:18:7c', '70:f9:27',

    # Xiaomi / Redmi
    'ac:c1:ee', 'f8:a4:5f', '28:6c:07', 'fc:64:ba', '00:9e:c8',
    '8c:be:be', '64:b4:73', 'a8:9a:93', '34:80:b3', '20:47:da',
    '58:44:98', '78:02:f8', 'c4:0b:cb', 'b0:e2:35', '68:df:dd',

    # OnePlus
    'ac:ed:5c', '94:65:2d', '30:45:96', '20:3d:b2', '08:f4:ab',

    # Realme / OPPO
    'c4:08:b0', '4c:49:e3', 'e0:46:9a', 'a8:93:4a', '18:26:49',
    '00:1a:ef', 'fc:3f:db', '84:a9:38', 'ac:37:43', '90:e7:c4',

    # Vivo
    '10:13:31', 'cc:79:cf', 'e4:a7:c5', '40:31:3c', '54:e0:61',
    '58:a2:c2', '00:d0:09', '28:5f:db', '9c:80:df', 'a4:50:46',

    # Huawei phones
    '00:18:82', '00:1e:10', '00:25:68', '04:bd:70', '08:19:a6',
    '0c:37:dc', '10:1b:54', '18:cf:5e', '1c:8e:5c', '20:0b:c7',
    '28:31:52', '2c:ab:00', '30:74:96', '34:29:12', '38:bc:01',
    '40:4d:8e', '44:6e:e5', '48:db:50', '4c:8b:ef', '50:a7:2b',

    # Motorola phones
    '00:08:a4', '00:1a:1b', '00:22:a4', '04:52:f3', '14:f6:5a',
    '2c:f0:a2', '34:bb:26', '40:b0:fa', '54:60:09', '58:40:4e',

    # Google Pixel
    '3c:28:6d', '64:bc:0c', '94:b4:0f', 'a4:77:33',
    'f4:f5:e8', '00:1a:11', '08:9e:08', '20:df:b9', '40:4e:36',
}

# ── FILTERS ───────────────────────────────────────────────────
def is_mobile_device(bssid):
    prefix = bssid[:8].lower()
    return prefix in MOBILE_OUI_PREFIXES

def is_stable_and_fixed(bssid, rssi_values, samples):
    if is_mobile_device(bssid):
        return False, "mobile MAC prefix"
    appearance_rate = len(rssi_values) / samples
    if appearance_rate < 0.7:
        return False, f"low appearance ({int(appearance_rate*100)}%)"
    if len(rssi_values) > 1:
        mean = sum(rssi_values) / len(rssi_values)
        variance = sum((v - mean) ** 2 for v in rssi_values) / len(rssi_values)
        if variance > 25:
            return False, f"high variance ({variance:.1f})"
    return True, "stable"

# ── SCAN ──────────────────────────────────────────────────────
def scan_wifi_windows():
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

# ── COLLECT ───────────────────────────────────────────────────
def collect_sample(location_name, samples=10):
    print(f"\nCollecting data for: {location_name}")
    print(f"Scanning WiFi {samples} times for stability analysis...")

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

    row = {'location': location_name, 'timestamp': datetime.now()}
    stable_count = 0
    unstable_count = 0

    print("\n  Filtering networks...")
    for key, data in all_networks.items():
        stable, reason = is_stable_and_fixed(key, data['rssi_values'], samples)
        if not stable:
            print(f"  [skip] {data['ssid']:<30} reason: {reason}")
            unstable_count += 1
            continue
        col_name = f"rssi_{data['ssid']}_{key[-5:]}"
        row[col_name] = round(sum(data['rssi_values']) / len(data['rssi_values']), 2)
        stable_count += 1
        print(f"  [keep] {data['ssid']:<30} avg: {row[col_name]} dBm")

    print(f"\n  Stable routers kept   : {stable_count}")
    print(f"  Unstable/mobile skipped: {unstable_count}")

    if stable_count == 0:
        print("  WARNING: No stable networks found at this location!")

    return row

# ── SAVE ──────────────────────────────────────────────────────
def save_to_csv(row, filepath=None):
    if filepath is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base, 'data', 'wifi_fingerprints.csv')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_new = pd.DataFrame([row])

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10:
        try:
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
            print(f"\n  Saved! Total rows in dataset: {len(df_combined)}")
        except Exception as e:
            print(f"  Warning: {e} — starting fresh")
            df_new.to_csv(filepath, index=False)
            print(f"\n  Saved! Total rows in dataset: 1")
    else:
        df_new.to_csv(filepath, index=False)
        print(f"\n  Saved! Total rows in dataset: 1")

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("   PathFi Data Collector — Dual Layer Filter")
    print("=" * 45)
    print("\nFiltering: MAC prefix (OUI) + Signal stability")
    print("Only fixed routers will be kept.\n")
    print("Location examples: lab, corridor, reception, canteen")
    print()

    location = input("Location name: ").strip().lower().replace(" ", "_")

    if not location:
        print("Location name cannot be empty!")
    else:
        row = collect_sample(location, samples=10)
        save_to_csv(row)
        print(f"\nDone! Data saved for '{location}'")
        print("Move to the next location and run the script again.")