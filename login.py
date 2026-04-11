# login.py  —  logs in and returns a working SmartAPI object

import pyotp
from SmartApi import SmartConnect
from config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET

def get_api():
    """Login to Angel One and return authenticated api object."""
    api = SmartConnect(api_key=API_KEY)
    
    # Generate the one-time password automatically from your secret
    totp_code = pyotp.TOTP(TOTP_SECRET).now()
    
    data = api.generateSession(CLIENT_ID, PASSWORD, totp_code)
    
    if data["status"] == False:
        raise Exception(f"Login failed: {data['message']}")
    
    expiry = data['data'].get('tokenExpiryTime', 'N/A')
    print(f"Login successful! Token expires at: {expiry}")
    return api

if __name__ == "__main__":
    api = get_api()
    print("Login test passed.")