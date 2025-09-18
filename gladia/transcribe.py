from dotenv import load_dotenv
import os
import logging

# .env laden
load_dotenv()

# Variablen abrufen
API_KEY = os.getenv("GLADIA_API_KEY")


import argparse
parser = argparse.ArgumentParser(description='Start Gladia Speech-to-Text test with configuration options.')
parser.add_argument('-l', '--lang', '--language', type=str, default="en",
                help='Language code for the STT model to transcribe in a specific language.')
parser.add_argument('--log', action='store_true', default=False)
args = parser.parse_args()