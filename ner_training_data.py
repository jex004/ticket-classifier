# ner_training_data.py

# Define the entities we want to extract
# SOFTWARE: Specific applications (Outlook, SAP, etc.)
# HARDWARE: Physical devices (laptop, printer, etc.)
# REQUEST_TYPE: The user's intent (access, reset, installation, etc.)

TRAIN_DATA = [
    ("My outlook is crashing", {"entities": [(3, 10, "SOFTWARE")]}),
    ("the vpn is not connecting", {"entities": [(4, 7, "SOFTWARE")]}),
    ("Cannot login to SAP", {"entities": [(16, 19, "SOFTWARE")]}),
    ("Need access to the shared drive", {"entities": [(5, 11, "REQUEST_TYPE")]}),
    ("My password needs a reset", {"entities": [(21, 26, "REQUEST_TYPE")]}),
    ("Requesting a new license for Adobe Photoshop", {"entities": [(12, 23, "REQUEST_TYPE"), (28, 43, "SOFTWARE")]}),
    ("The printer is not working", {"entities": [(4, 11, "HARDWARE")]}),
    ("My laptop screen is broken", {"entities": [(3, 9, "HARDWARE")]}),
    ("The keyboard is not responding", {"entities": [(4, 12, "HARDWARE")]}),
    ("I need a new mouse", {"entities": [(13, 18, "HARDWARE")]}),
    ("Please install Google Chrome on my machine", {"entities": [(7, 14, "REQUEST_TYPE"), (15, 28, "SOFTWARE")]}),
    ("My wifi is down", {"entities": [(3, 7, "HARDWARE")]}),
    ("Blue screen on my Dell computer", {"entities": [(21, 34, "HARDWARE")]}),
    ("Can I get a new monitor?", {"entities": [(17, 24, "HARDWARE")]}),
    ("I am unable to access my email", {"entities": [(21, 27, "REQUEST_TYPE"), (28, 33, "SOFTWARE")]}),
    ("My phone is not syncing with Teams", {"entities": [(3, 8, "HARDWARE"), (30, 35, "SOFTWARE")]}),
    ("Request for admin rights", {"entities": [(12, 24, "REQUEST_TYPE")]}),
    ("Failure to connect to the server", {"entities": [(28, 34, "HARDWARE")]}),
    ("Software update for windows failed", {"entities": [(0, 16, "REQUEST_TYPE"), (21, 28, "SOFTWARE")]}),
    ("My webcam is not detected in Zoom", {"entities": [(3, 9, "HARDWARE"), (30, 34, "SOFTWARE")]}),
]