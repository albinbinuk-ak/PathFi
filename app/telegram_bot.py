from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ParseMode
import subprocess
import pandas as pd
import pickle
import logging

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ── YOUR BOT TOKEN HERE ───────────────────────────────────────
TOKEN = "8891478552:AAFICDWwRAkj8kKLDNNiNgkmHa3qF-K-JJI"

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

# ── COMMANDS ──────────────────────────────────────────────────
def start(update, context):
    update.message.reply_text(
        "👋 Welcome to *PathFi* — Intelligent Indoor Navigator!\n\n"
        "Commands:\n"
        "📍 /locate — Predict your current location\n"
        "📡 /signals — Show visible WiFi networks\n"
        "ℹ️ /about — About PathFi\n",
        parse_mode=ParseMode.MARKDOWN
    )

def locate(update, context):
    update.message.reply_text("📡 Scanning WiFi networks...")
    
    try:
        model, le, feature_columns = load_model()
        networks = scan_wifi()
        
        if not networks:
            update.message.reply_text("❌ No WiFi networks detected. Make sure WiFi is on.")
            return
        
        # Build feature row
        row = {col: networks.get(col, -100) for col in feature_columns}
        X_input = pd.DataFrame([row])
        
        prediction = model.predict(X_input)[0]
        location = le.inverse_transform([prediction])[0]
        location_display = location.replace('_', ' ').title()
        
        # Signal summary
        visible = {k.split('_')[1]: v for k, v in networks.items()}
        signal_lines = '\n'.join([f"  • {n}: `{v:.1f} dBm`" for n, v in list(visible.items())[:5]])
        
        update.message.reply_text(
            f"📍 *You are at: {location_display}*\n\n"
            f"*Visible networks:*\n{signal_lines}\n\n"
            f"_Powered by PathFi ML Model_",
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

def signals(update, context):
    update.message.reply_text("📡 Scanning...")
    
    networks = scan_wifi()
    
    if not networks:
        update.message.reply_text("No networks found.")
        return
    
    lines = [f"• {k.split('_')[1]}: `{v:.1f} dBm`" for k, v in networks.items()]
    update.message.reply_text(
        f"*Visible WiFi Networks:*\n\n" + '\n'.join(lines),
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
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("locate", locate))
    dp.add_handler(CommandHandler("signals", signals))
    dp.add_handler(CommandHandler("about", about))
    
    print("PathFi Bot is running... Press Ctrl+C to stop")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()