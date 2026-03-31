"""
Enable the Gmail API
Before running the code, you need to register your "App" (the script) with Google:
1. Go to the Google Cloud Console.
2. Create a new project (e.g., "DGX-CECL-Reports").
3. Navigate to APIs & Services > Library and search for "Gmail API"—click Enable.
4. Go to OAuth consent screen, select External, and add your email as a Test User.
5. Go to Credentials, click Create Credentials > OAuth client ID, and select Desktop App.
6. Download the JSON file, rename it to credentials.json, and place it in your script folder.

Runs the Fannie Mae pipeline every Monday at 1:00 AM
0 1 * * 1 /usr/bin/python3 /home/user/run_cecl_pipeline.py --data_dir /data/fannie --gse fannie && /usr/bin/python3 /home/user/send_report.py
"""

import os
import base64
import pandas as pd
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def send_cecl_report(results_path, recipient):
    # 1. Generate Summary from DGX Results
    df = pd.read_parquet(results_path)
    total_upb = df['current_upb'].sum()
    total_ecl = df['expected_loss'].sum()
    
    # 2. Build the Email
    msg = EmailMessage()
    msg.set_content(f"CECL Run Complete.\nTotal UPB: ${total_upb:,.2f}\nExpected Loss: ${total_ecl:,.2f}")
    msg['Subject'] = '🚀 DGX Spark: CECL Portfolio Update'
    msg['To'] = recipient
    msg['From'] = 'me'

    # 3. Encode and Send via API
    service = get_gmail_service()
    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    create_message = {'raw': encoded_message}
    
    try:
        send_message = (service.users().messages().send(userId="me", body=create_message).execute())
        print(f'✅ Email Sent! Message ID: {send_message["id"]}')
    except Exception as e:
        print(f'❌ An error occurred: {e}')

if __name__ == "__main__":
    send_cecl_report('fannie_results.parquet', 'your-email@gmail.com')

