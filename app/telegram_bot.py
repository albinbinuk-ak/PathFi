from telegram.ext import Updater, CommandHandler
from telegram import ParseMode
import subprocess
import pandas as pd
import pickle
import logging
import heapq
from location_log import log_location, get_log, get_today_summary, get_last_location

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ── YOUR BOT TOKEN ────────────────────────────────────────────
TOKEN = "8891478552:AAFICDWwRAkj8kKLDNNiNgkmHa3qF-K-JJI"

# ── LOCATION GRAPH ────────────────────────────────────────────
def get_locations_from_data():
    """Auto-generate location graph and labels from CSV"""
    try:
        df = pd.read_csv('data/wifi_fingerprints.csv')
        locations = df['location'].unique().tolist()
    except Exception:
        locations = []

    # Auto-generate labels (convert underscore to title case)
    labels = {loc: loc.replace('_', ' ').title() for loc in locations}

    # Auto-generate graph (connect every location to every other)
    # Default distance 10m between all — user can tune later
    graph = {}
    for loc in locations:
        graph[loc] = {}
        for other in locations:
            if other != loc:
                graph[loc][other] = 10  # default 10m

    # Auto-generate directions
    directions = {}
    for loc in locations:
        for other in locations:
            if other != loc:
                directions[(loc, other)] = f"Walk from {labels.get(loc, loc)} toward {labels.get(other, other)}"

    return graph, labels, directions

LOCATION_GRAPH, LOCATION_LABELS, DIRECTIONS = get_locations_from_data()

# ── DIJKSTRA ──────────────────────────────────────────────────
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            return path, cost
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return None, float('inf')

# ── LOAD MODEL ────────────────────────────────────────────────
def load_model():
    with open('model/pathfi_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('model/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, features

# ── WIFI SCAN ─────────────────────────────────────────────────
def scan_wifi():
    result = subprocess.run(
        ["netsh", "wlan", "show", "networks", "mode=bssid"],
        capture_output=True, text=True
    )
    networks = {}
    lines = result.stdout.split('\n')
    current_ssid = None
    current_bssid = None

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
            if len(parts) > 1 and current_ssid and current_bssid:
                signal_percent = int(parts[1].strip().replace('%', ''))
                rssi = (signal_percent / 2) - 100
                col_name = f"rssi_{current_ssid}_{current_bssid[-5:]}"
                networks[col_name] = rssi
                current_ssid = None
                current_bssid = None
    return networks

# ── PREDICT HELPER ────────────────────────────────────────────
def predict_current_location():
    model, le, feature_columns = load_model()
    networks = scan_wifi()
    row = {col: networks.get(col, -100) for col in feature_columns}
    X_input = pd.DataFrame([row])

    prediction = model.predict(X_input)[0]
    location = le.inverse_transform([prediction])[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        confidence = round(max(proba) * 100, 1)
    else:
        confidence = None

    return location, confidence, networks

# ── COMMANDS ──────────────────────────────────────────────────
def start(update, context):
    update.message.reply_text(
        "👋 Welcome to *PathFi* — Intelligent Indoor Navigator!\n\n"
        "Commands:\n"
        "📍 /locate — Predict your current location\n"
        "🗺️ /navigate <location> — Get route to destination\n"
        "📊 /summary — Today's presence summary\n"
        "📡 /signals — Show visible WiFi networks\n"
        "ℹ️ /about — About PathFi\n",
        parse_mode=ParseMode.MARKDOWN
    )

def locate(update, context):
    update.message.reply_text("📡 Scanning WiFi networks...")
    try:
        location, confidence, networks = predict_current_location()
        log_location(location)

        display = LOCATION_LABELS.get(location, location.replace('_', ' ').title())
        conf_text = f" ({confidence}% confident)" if confidence else ""

        if confidence and confidence < 60:
            status = f"⚠️ Uncertain — closest match: *{display}*{conf_text}"
        else:
            status = f"📍 You are at: *{display}*{conf_text}"

        visible = {k: v for k, v in networks.items() if v != -100}
        signal_lines = '\n'.join([
            f"  • {k.split('_')[1]}: `{v:.1f} dBm`"
            for k, v in list(visible.items())[:5]
        ])

        update.message.reply_text(
            f"{status}\n\n"
            f"*Visible networks:*\n{signal_lines}\n\n"
            f"_Powered by PathFi ML Model_",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

def navigate(update, context):
    args = context.args
    if len(args) < 1:
        locs = '\n'.join([f"  • `{k}` → {v}" for k, v in LOCATION_LABELS.items()])
        update.message.reply_text(
            "*Usage:* /navigate <destination>\n\n"
            f"*Available locations:*\n{locs}\n\n"
            "_Example:_ /navigate main\\_door",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    dest = args[0].lower()
    if dest not in LOCATION_GRAPH:
        locs = '\n'.join([f"  • `{k}`" for k in LOCATION_GRAPH.keys()])
        update.message.reply_text(
            f"❌ Unknown location: `{dest}`\n\n"
            f"*Available:*\n{locs}",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    update.message.reply_text("📡 Detecting your current location...")

    try:
        location, confidence, _ = predict_current_location()
        log_location(location)
        current = location

        if current == dest:
            update.message.reply_text(
                f"✅ You are already at *{LOCATION_LABELS.get(dest, dest)}*!",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        path, distance = dijkstra(LOCATION_GRAPH, current, dest)

        if not path:
            update.message.reply_text("❌ No route found between these locations.")
            return

        steps = []
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i+1]
            src_label = LOCATION_LABELS.get(src, src)
            dst_label = LOCATION_LABELS.get(dst, dst)
            d = LOCATION_GRAPH[src][dst]
            direction = DIRECTIONS.get((src, dst), "Walk toward the location")
            steps.append(f"{i+1}. {direction} (~{d}m) → **{dst_label}**")

        route_display = ' → '.join([LOCATION_LABELS.get(p, p) for p in path])
        conf_text = f" ({confidence}% confident)" if confidence else ""

        update.message.reply_text(
            f"🗺️ *Navigation Started*\n\n"
            f"📍 From: *{LOCATION_LABELS.get(current, current)}*{conf_text}\n"
            f"🏁 To: *{LOCATION_LABELS.get(dest, dest)}*\n\n"
            f"*Route:*\n{route_display}\n\n"
            f"*Step-by-step:*\n" + '\n'.join(steps) + f"\n\n"
            f"📏 Total: *{distance}m* • {len(path)-1} step{'s' if len(path)>2 else ''}",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

def summary(update, context):
    counts = get_today_summary()
    if not counts:
        update.message.reply_text(
            "📊 No location data for today yet.\nUse /locate first!"
        )
        return

    lines = [
        f"• {LOCATION_LABELS.get(loc, loc.replace('_',' ').title())}: {duration}"
        
        for loc, duration in counts.items()
    ]
    total_scans = len(get_log())

    update.message.reply_text(
        f"📊 *Today's Presence Summary:*\n\n" +
        '\n'.join(lines) +
        f"\n\n_Scan more frequently for accurate time tracking_",
        parse_mode=ParseMode.MARKDOWN
    )

def signals(update, context):
    update.message.reply_text("📡 Scanning...")
    networks = scan_wifi()
    if not networks:
        update.message.reply_text("No networks found.")
        return
    lines = [
        f"• {k.split('_')[1]}: `{v:.1f} dBm`"
        for k, v in networks.items()
    ]
    update.message.reply_text(
        "*Visible WiFi Networks:*\n\n" + '\n'.join(lines),
        parse_mode=ParseMode.MARKDOWN
    )

def about(update, context):
    update.message.reply_text(
        "*PathFi — Intelligent Indoor Navigator*\n\n"
        "Uses WiFi signal fingerprinting and Machine Learning "
        "to predict your indoor location without GPS.\n\n"
        "*Tech Stack:*\n"
        "• Python + scikit-learn\n"
        "• Random Forest Classifier\n"
        "• Dijkstra's Shortest Path Algorithm\n"
        "• Streamlit Dashboard\n"
        "• Telegram Bot Interface\n\n"
        "*Inspired by LoRa RSSI-based asset tracking*\n\n"
        "_Built by an Intern who skipped ML classes_ 😄",
        parse_mode=ParseMode.MARKDOWN
    )

# ── MAIN ──────────────────────────────────────────────────────
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start",    start))
    dp.add_handler(CommandHandler("locate",   locate))
    dp.add_handler(CommandHandler("navigate", navigate))
    dp.add_handler(CommandHandler("summary",  summary))
    dp.add_handler(CommandHandler("signals",  signals))
    dp.add_handler(CommandHandler("about",    about))

    print("🚀 PathFi Bot is running... Press Ctrl+C to stop")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()