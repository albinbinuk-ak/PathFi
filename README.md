# PathFi

Intelligent Indoor Navigator using WiFi RSSI fingerprinting and Machine Learning


\# PathFi 📍

\### Intelligent Indoor Navigator



> WiFi RSSI fingerprinting + Machine Learning + Streamlit dashboard + Telegram bot



\---



\## What is PathFi?



GPS doesn't work indoors. PathFi solves this by using the WiFi signals already around you — every router and access point broadcasts a signal strength (RSSI) that changes depending on where you are. PathFi collects these signal patterns at known locations, trains a machine learning model to recognise them, and then predicts your location in real time.



No extra hardware. No GPS. Just your laptop and the WiFi that's already there.



\---



\## How it works



```

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

```



\---



\## Tech Stack



| Layer | Technology |

|---|---|

| Data collection | Python, scapy / subprocess |

| Data processing | pandas, numpy |

| Machine learning | scikit-learn |

| Dashboard | Streamlit |

| Bot interface | python-telegram-bot |

| Version control | Git + GitHub |



\---



\## Project Structure



```

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

```



\---



\## ML Models Used



| Model | Why we use it |

|---|---|

| K-Nearest Neighbors | Works naturally for location — nearby signal patterns = nearby location |

| Random Forest | Handles noisy WiFi data well, good accuracy |

| Support Vector Machine | Strong baseline for multi-class classification |



The best performing model is saved and used for real-time prediction.



\---



\## Results



> \*(To be updated after training)\*



| Model | Accuracy |

|---|---|

| KNN | — |

| Random Forest | — |

| SVM | — |



\---



\## Real World Connection



This project is inspired by RSSI-based positioning used in \*\*LoRa IoT networks\*\* for industrial asset tracking. The same principle — signal strength fingerprinting — is how LoRa gateways track vehicles and workers in factories and logistics yards. PathFi demonstrates this concept using existing WiFi infrastructure, making it deployable in any building without additional hardware.



\---



\## Applications



\- Smart campus navigation

\- Indoor asset and equipment tracking

\- Occupancy monitoring for smart buildings

\- Accessibility assistance for visually impaired users



\---



\## Setup



```bash

git clone https://github.com/YOUR-USERNAME/PathFi.git

cd PathFi

pip install -r requirements.txt

```



Run the dashboard:

```bash

streamlit run app/dashboard.py

```



Run the Telegram bot:

```bash

python app/telegram\_bot.py

```



\---



\## Progress



\- \[x] Project finalised and abstract submitted

\- \[ ] GitHub repo set up

\- \[ ] Data collection script ready

\- \[ ] WiFi fingerprint data collected

\- \[ ] ML models trained and compared

\- \[ ] Streamlit dashboard built

\- \[ ] Telegram bot connected

\- \[ ] Live demo ready



\---



\## Built By



\*\*Albin Binu K\*\*

BCA \[Marian Academy Of Management Studies Kothamangalam]

S5 C



\---



\*PathFi — because knowing where you are shouldn't need satellites.\*





