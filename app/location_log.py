import pandas as pd
import os
from datetime import datetime

LOG_FILE = 'data/location_log.csv'

def log_location(location):
    """Save a location detection event to log"""
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'location': location
    }
    df_new = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df_existing = pd.read_csv(LOG_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(LOG_FILE, index=False)
    else:
        df_new.to_csv(LOG_FILE, index=False)

def get_log():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame(columns=['timestamp', 'date', 'time', 'location'])
            return df
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'date', 'time', 'location'])
    return pd.DataFrame(columns=['timestamp', 'date', 'time', 'location'])

def get_today_summary():
    """Return today's time spent per location"""
    df = get_log()
    if df.empty:
        return {}
    
    today = datetime.now().strftime('%Y-%m-%d')
    df_today = df[df['date'] == today].copy()
    if df_today.empty:
        return {}
    
    df_today['timestamp'] = pd.to_datetime(df_today['timestamp'])
    df_today = df_today.sort_values('timestamp').reset_index(drop=True)
    
    time_spent = {}
    
    for i in range(len(df_today) - 1):
        loc = df_today.loc[i, 'location']
        t1 = df_today.loc[i, 'timestamp']
        t2 = df_today.loc[i + 1, 'timestamp']
        gap = (t2 - t1).total_seconds() / 60  # in minutes
        
        # Only count gaps under 30 mins (avoids counting idle time)
        if gap < 30:
            time_spent[loc] = time_spent.get(loc, 0) + gap
    
    # Add last location with 1 min minimum
    last_loc = df_today.iloc[-1]['location']
    time_spent[last_loc] = time_spent.get(last_loc, 0) + 1
    
    # Convert to readable format
    result = {}
    for loc, mins in time_spent.items():
        if mins >= 60:
            hrs = int(mins // 60)
            rem = int(mins % 60)
            result[loc] = f"{hrs}hr {rem}min" if rem > 0 else f"{hrs}hr"
        else:
            result[loc] = f"{int(mins)}min"
    
    return result

def get_last_location():
    """Return the most recent logged location"""
    df = get_log()
    if df.empty:
        return None
    return df.iloc[-1]['location']