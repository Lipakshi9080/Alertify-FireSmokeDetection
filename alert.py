"""
alert.py — Hazard Alert System
================================
Handles all outbound notifications when fire or smoke is detected:
  • SMS alerts via Twilio REST API
  • Email alerts via Gmail SMTP (smtplib)

Design pattern
--------------
Both functions are intentionally standalone so they can be called
independently from detect.py, tested in isolation, or swapped out
for other providers (e.g. SendGrid, AWS SNS) without touching the
rest of the codebase.
"""

import smtplib
import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# ── Twilio SDK ───────────────────────────────────────────────
#try:
 #   from twilio.rest import Client as TwilioClient
  #except ImportError:
   # TWILIO_AVAILABLE = False
    #print("[WARN] twilio package not installed – SMS alerts disabled.")

# ── Load credentials from environment / .env file ───────────
load_dotenv()  # reads .env in project root automatically

# ── Twilio credentials ───────────────────────────────────────
#TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "YOUR_TWILIO_SID")
#TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",  "+1XXXXXXXXXX")
#ALERT_TO_NUMBER     = os.getenv("ALERT_TO_NUMBER",     "+91XXXXXXXXXX")

# ── Gmail SMTP credentials ───────────────────────────────────
GMAIL_ADDRESS       = os.getenv("GMAIL_ADDRESS",       "your_email@gmail.com")
GMAIL_APP_PASSWORD  = os.getenv("GMAIL_APP_PASSWORD",  "your_app_password")
ALERT_TO_EMAIL      = os.getenv("ALERT_TO_EMAIL",      "recipient@example.com")


# ════════════════════════════════════════════════════════════════
# SMS Alert  (Twilio)
# ════════════════════════════════════════════════════════════════

def send_sms_alert(hazard_type: str, confidence: float, screenshot_path: str) -> bool:
    """
    Sends an SMS message to the configured phone number via Twilio.

    Parameters
    ----------
    hazard_type     : "Fire" | "Smoke" | "Fire & Smoke"
    confidence      : detection confidence as a float  0.0 – 1.0
    screenshot_path : path to the saved screenshot file

    Returns True on success, False on failure.
    """

    if not TWILIO_AVAILABLE:
        print("[SMS] Twilio not installed – skipping SMS alert.")
        return False

    # Guard: skip if placeholder credentials are still in place
    if "YOUR_TWILIO" in TWILIO_ACCOUNT_SID or "YOUR_TWILIO" in TWILIO_AUTH_TOKEN:
        print("[SMS] Twilio credentials not configured in .env – skipping.")
        return False

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        body = (
            f"🔥 FIRE/SMOKE ALERT 🔥\n"
            f"Hazard Detected : {hazard_type}\n"
            f"Confidence      : {confidence:.1%}\n"
            f"Time            : {timestamp}\n"
            f"Screenshot      : {screenshot_path}\n\n"
            f"Please check your surveillance feed immediately!"
        )

        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER,
        )

        print(f"[SMS] Alert sent ✓  SID={message.sid}")
        return True

    except Exception as exc:
        print(f"[SMS] Failed to send alert: {exc}")
        return False


# ════════════════════════════════════════════════════════════════
# Email Alert  (Gmail SMTP)
# ════════════════════════════════════════════════════════════════

def send_email_alert(hazard_type: str, confidence: float, screenshot_path: str) -> bool:
    """
    Sends an HTML email with an optional screenshot attachment via Gmail SMTP.

    Parameters
    ----------
    hazard_type     : "Fire" | "Smoke" | "Fire & Smoke"
    confidence      : detection confidence as a float  0.0 – 1.0
    screenshot_path : path to the saved screenshot file

    Returns True on success, False on failure.
    """

    # Guard: skip if placeholder credentials are still in place
    if "your_email" in GMAIL_ADDRESS or "your_app" in GMAIL_APP_PASSWORD:
        print("[EMAIL] Gmail credentials not configured in .env – skipping.")
        return False

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Build the email structure ────────────────────────
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔥 FIRE/SMOKE ALERT — {hazard_type} Detected!"
        msg["From"]    = GMAIL_ADDRESS
        receivers = [email.strip() for email in ALERT_TO_EMAIL.split(",")]
        msg["To"] = ", ".join(receivers)
        # Plain-text fallback
        plain_text = (
            f"HAZARD DETECTED: {hazard_type}\n"
            f"Confidence : {confidence:.1%}\n"
            f"Timestamp  : {timestamp}\n"
            f"Screenshot : {screenshot_path}\n"
        )

        # Rich HTML body
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:24px;">
          <div style="max-width:600px;margin:auto;border:2px solid #ff4500;border-radius:12px;overflow:hidden;">
            <div style="background:#ff4500;padding:20px;text-align:center;">
              <h1 style="margin:0;color:#fff;font-size:28px;">🔥 FIRE / SMOKE ALERT</h1>
            </div>
            <div style="padding:24px;background:#1a1a1a;">
              <table style="width:100%;border-collapse:collapse;">
                <tr>
                  <td style="padding:10px;color:#aaa;width:160px;">Hazard Type</td>
                  <td style="padding:10px;color:#ff6b35;font-weight:bold;font-size:18px;">{hazard_type}</td>
                </tr>
                <tr style="background:#222;">
                  <td style="padding:10px;color:#aaa;">Confidence</td>
                  <td style="padding:10px;color:#fff;">{confidence:.1%}</td>
                </tr>
                <tr>
                  <td style="padding:10px;color:#aaa;">Timestamp</td>
                  <td style="padding:10px;color:#fff;">{timestamp}</td>
                </tr>
                <tr style="background:#222;">
                  <td style="padding:10px;color:#aaa;">Screenshot</td>
                  <td style="padding:10px;color:#fff;">{screenshot_path}</td>
                </tr>
              </table>
              <p style="margin-top:20px;color:#ccc;">
                ⚠️ Please check your surveillance system immediately and take appropriate action.
              </p>
            </div>
            <div style="background:#0d0d0d;padding:12px;text-align:center;color:#555;font-size:12px;">
              FireSmokeDetection AI System · Automated Safety Alert
            </div>
          </div>
        </body></html>
        """

        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_body,  "html"))

        # ── Attach screenshot if it exists ───────────────────
        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            fname = os.path.basename(screenshot_path)
            part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
            msg.attach(part)

        # ── Connect and send ─────────────────────────────────
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, receivers, msg.as_string())
        print(f"[EMAIL] Alert sent ✓  To={ALERT_TO_EMAIL}")
        return True

    except Exception as exc:
        print(f"[EMAIL] Failed to send alert: {exc}")
        return False


# ════════════════════════════════════════════════════════════════
# Unified dispatcher — calls both SMS + Email
# ════════════════════════════════════════════════════════════════

def dispatch_alerts(hazard_type: str, confidence: float, screenshot_path: str) -> None:
    """
    Convenience wrapper that fires both SMS and email alerts in sequence.
    Called by detect.py whenever the cooldown timer allows a new alert.
    """
    print(f"\n[ALERT] Dispatching alerts for '{hazard_type}' ({confidence:.1%}) …")
    #send_sms_alert(hazard_type, confidence, screenshot_path)
    send_email_alert(hazard_type, confidence, screenshot_path)
    print("[ALERT] Alert dispatch complete.\n")
