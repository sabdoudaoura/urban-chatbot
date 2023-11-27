import os
from dotenv import load_dotenv
from chromadb.config import Settings

# Load variables from .env into environment
load_dotenv()

# we create out database in the persist directory
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if PERSIST_DIRECTORY is None:
    raise Exception("Please set the PERSIST_DIRECTORY environment variable")


CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


