# PathFi

Intelligent Indoor Navigator using WiFi RSSI fingerprinting and Machine Learning




WiFi RSSI fingerprinting + Machine Learning + Streamlit dashboard + Telegram bot





What is PathFi?

GPS doesn't work indoors. PathFi solves this by using the WiFi signals already around you — every router and access point broadcasts a signal strength (RSSI) that changes depending on where you are. PathFi collects these signal patterns at known locations, trains a machine learning model to recognise them, and then predicts your location in real time.

No extra hardware. No GPS. Just your laptop and the WiFi that's already there.



How it works

Your Laptop

&#x20;   │

&#x20;   ├── Scans nearby WiFi networks → records signal strengths (RSSI)

&#x20;   │

&#x20;   ├── ML Model (trained on your building's WiFi fingerprint)

&#x20;   │       ├── K-Nearest Neighbors

&#x20;   │       ├── Random Forest

&#x20;   │       └── Support Vector Machine

&#x20;   │

&#x20;   ├── Streamlit Dashboard → live location + accuracy charts

&#x20;   │

&#x20;   └── Telegram Bot → /locate → "You are at: Computer Lab"



Tech Stack

LayerTechnologyData collectionPython, scapy / subprocessData processingpandas, numpyMachine learningscikit-learnDashboardStreamlitBot interfacepython-telegram-botVersion controlGit + GitHub



Project Structure

PathFi/

│

├── data/

│   └── wifi\_fingerprints.csv      # Collected RSSI data with location labels

│

├── model/

│   └── train\_model.py             # Train and compare ML models

│   └── pathfi\_model.pkl           # Saved best model

│

├── app/

│   └── dashboard.py               # Streamlit web dashboard

│   └── telegram\_bot.py            # Telegram bot interface

│

├── collect/

│   └── scan\_wifi.py               # Data collection script

│

├── requirements.txt

└── README.md



ML Models Used

ModelWhy we use itK-Nearest NeighborsWorks naturally for location — nearby signal patterns = nearby locationRandom ForestHandles noisy WiFi data well, good accuracySupport Vector MachineStrong baseline for multi-class classification

The best performing model is saved and used for real-time prediction.



Results



(To be updated after training)



ModelAccuracyKNN—Random Forest—SVM—



Real World Connection

This project is inspired by RSSI-based positioning used in LoRa IoT networks for industrial asset tracking. The same principle — signal strength fingerprinting — is how LoRa gateways track vehicles and workers in factories and logistics yards. PathFi demonstrates this concept using existing WiFi infrastructure, making it deployable in any building without additional hardware.



Applications



Smart campus navigation

Indoor asset and equipment tracking

Occupancy monitoring for smart buildings

Accessibility assistance for visually impaired users





Setup

bashgit clone https://github.com/YOUR-USERNAME/PathFi.git

cd PathFi

pip install -r requirements.txt

Run the dashboard:

bashstreamlit run app/dashboard.py

Run the Telegram bot:

bashpython app/telegram\_bot.py



Progress



&#x20;Project finalised and abstract submitted

&#x20;GitHub repo set up

&#x20;Data collection script ready

&#x20;WiFi fingerprint data collected

&#x20;ML models trained and compared

&#x20;Streamlit dashboard built

&#x20;Telegram bot connected

&#x20;Live demo ready





Built By

Albin Binu K

BCA S5 C



PathFi — because knowing where you are shouldn't need satellites.

