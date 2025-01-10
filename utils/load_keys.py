from dotenv import load_dotenv
import os

def load_secrets():
    """
    Load secrets from a .env file.

    Returns:
    - dict: A dictionary containing the secrets.
    """
    load_dotenv()

    return {
        "rest_api_key": os.getenv("REST_API_KEY"),
        "rest_api_secret": os.getenv("REST_API_SECRET"),
    }
