# email_alerts.py
import smtplib
import os
import pathlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_critical_report_email(report_path: str = "execution_report.html"):
    """
    Connects to the operations SMTP relay to forward the HTML report card 
    directly to the dispatch inbox following a critical threshold incident.
    """
    # 1. CONFIGURE PRODUCTION SMTP ROUTING ENVS
    SMTP_SERVER = os.getenv("UNIVAC_SMTP_SERVER", "://your-operations-relay.com")
    SMTP_PORT = int(os.getenv("UNIVAC_SMTP_PORT", "587"))
    SMTP_USER = os.getenv("UNIVAC_SMTP_USER", "dispatch-alerts@univac-ix.org")
    SMTP_PASS = os.getenv("UNIVAC_SMTP_PASSWORD", "YourSecureRelayTokenSecret")
    DISPATCH_INBOX = os.getenv("UNIVAC_DISPATCH_INBOX", "ops-dispatch@univac-ix.org")

    report_file = pathlib.Path(report_path)
    if not report_file.exists():
        print(f"[-] Email Transmission Aborted: Target report '{report_path}' is missing from the directory tree.")
        return False

    print(f"[EMAIL-ALERT] Assembling critical incident message for dispatch: {DISPATCH_INBOX}")

    # 2. CONSTRUCT MULTI-PART MIME MESSAGE STRUCTURE
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = DISPATCH_INBOX
    msg['Subject'] = "🚨 CRITICAL THRESHOLD ALERT: UNIVAC-IX Node Core Exception Event"

    # Define the plaintext notification body
    body = (
        "WARNING: The UNIVAC-IX core fabric has identified an out-of-bounds anomaly threshold.\n"
        "The automated fallback recovery modules have triggered state stabilization cycles.\n\n"
        "The current system state ledger has been compiled and is attached below as an HTML report.\n"
        "Please review the active Visio data graphics stream immediately.\n"
    )
    msg.attach(MIMEText(body, 'plain'))

    # 3. ATTACH THE EXTRACTED HTML EXECUTION REPORT CARD
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            attachment_payload = MIMEText(f.read(), 'html')
            
        attachment_payload.add_header(
            "Content-Disposition",
            f"attachment; filename={report_file.name}"
        )
        msg.attach(attachment_payload)
        
    except Exception as err:
        print(f"[-] Failed to read and package HTML attachment structure: {str(err)}")
        return False

    # 4. EXECUTE SMTP TRANSIT HANDSHAKE WITH STARTTLS
    try:
        print(f"[EMAIL-ALERT] Establishing handshake connection to {SMTP_SERVER}:{SMTP_PORT}...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.ehlo()
        server.starttls()  # Force cryptographic transmission line encryption
        server.ehlo()
        
        # Authenticate against the industrial messaging subsystem channels
        server.login(SMTP_USER, SMTP_PASS)
        
        # Broadcast across the line wires
        server.sendmail(SMTP_USER, DISPATCH_INBOX, msg.as_string())
        server.quit()
        
        print(" ✓ Email transmission complete. Message accepted by dispatch mail exchange server.")
        return True
        
    except Exception as connection_error:
        print(f"[-] SMTP Relay Transmission Failure: {str(connection_error)}")
        print(" ⚠️  [LOG CHECK]: Ensure environmental credentials for the network are populated before running pipelines.")
        return False

if __name__ == "__main__":
    # Standalone script execution test (Uses mock file configuration)
    mock_file = "execution_report.html"
    if not os.path.exists(mock_file):
        with open(mock_file, "w") as f_mock:
            f_mock.write("<h1>Mock Operational Report</h1>")
            
    send_critical_report_email(report_path=mock_file)
