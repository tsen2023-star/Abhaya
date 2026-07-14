import logging

# Set up basic logging to see our simulated alerts in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_emergency_sms(user_id: str, lat: float, lon: float):
    """
    Simulates sending an SMS to emergency contacts.
    In production, you will replace the logger with a Twilio API or local SMS gateway call.
    """
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    message = f"URGENT: Abhaya user {user_id} triggered an SOS! Last known location: {maps_link}"
    
    # Log to terminal to verify it works locally
    logger.info(f"========== [SMS ALERT DISPATCHED] ==========")
    logger.info(message)
    logger.info(f"============================================")
    
    return True

def trigger_fake_call_logic(user_id: str):
    """
    Simulates triggering a VoIP call to the user's phone to help them escape a situation.
    In production, this hooks into Twilio Programmable Voice or a Firebase Push Notification.
    """
    logger.info(f"========== [FAKE CALL INITIATED] ==========")
    logger.info(f"Dialing user {user_id} to provide an exit excuse...")
    logger.info(f"===========================================")
    
    return True