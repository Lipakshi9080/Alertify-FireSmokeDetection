# 🔥 Alertify-FireSmokeDetection

Real-time AI-powered fire and smoke detection system using **YOLOv8**, **OpenCV**, and automated **email alerting**.


## 🚀 Features

* 🔥 Real-time fire detection
* 🌫 Smoke detection using YOLOv8
* 📹 Live webcam/CCTV feed monitoring
* 📸 Automatic screenshot capture
* 📧 Multi-recipient email alerts
* ⚡ Confidence threshold filtering
* 🧠 AI-powered object detection
* 💾 Annotated video output saving
* 🛡 Alert cooldown system to reduce spam alerts


## 🛠 Tech Stack

* Python
* YOLOv8
* OpenCV
* Ultralytics
* SMTP (Gmail)
* dotenv


## 🧠 System Architecture

```text
Webcam / CCTV Feed
        ↓
OpenCV Frame Capture
        ↓
YOLOv8 Inference
        ↓
Fire / Smoke Detection
        ↓
Confidence Filtering
        ↓
Screenshot Capture
        ↓
Email Alert Dispatch
```
## 📂 Project Structure

```text
Alertify-FireSmokeDetection/
│
├── models/
├── screenshots/
├── output/
├── main.py
├── detect.py
├── alert.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
```


## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Alertify-FireSmokeDetection.git
```

### 2️⃣ Move Into Project Folder

```bash
cd Alertify-FireSmokeDetection
```

### 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 4️⃣ Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 📧 Environment Variables

Create a `.env` file:

```env
GMAIL_ADDRESS=yourgmail@gmail.com
GMAIL_APP_PASSWORD=your_app_password

ALERT_TO_EMAIL=mail1@gmail.com,mail2@gmail.com
```
## ▶️ Run Project

```bash
python main.py
```
## 📸 Output

When fire or smoke is detected:

* Screenshot is captured automatically
* Email alert is sent
* Detection confidence is displayed
* Output video is saved

## 🔮 Future Scope

* Telegram / WhatsApp alerts
* Cloud deployment
* Multi-camera support
* Industrial IoT integration
* Edge AI deployment
* Mobile application support

## 👨‍💻 Author

Lipakshi Maurya

B.Tech CSE (AI)
GCET
