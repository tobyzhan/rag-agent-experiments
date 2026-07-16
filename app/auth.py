from itsdangerous import URLSafeTimedSerializer, BadSignature
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import utils

_serializer = URLSafeTimedSerializer(utils.get_session_secret_key())
COOKIE_NAME = "session"
COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days


def verify_google_token(credential: str) -> dict:
    """Verifies a Google ID token and returns {sub, email, name, picture}."""
    info = id_token.verify_oauth2_token(
        credential, google_requests.Request(), utils.get_google_client_id()
    )
    return {
        "sub": info["sub"],
        "email": info["email"],
        "name": info.get("name", info["email"]),
        "picture": info.get("picture", ""),
    }


def create_session_cookie_value(user: dict) -> str:
    return _serializer.dumps(user)


def read_session_cookie_value(cookie_value: str) -> dict | None:
    try:
        return _serializer.loads(cookie_value, max_age=COOKIE_MAX_AGE_SECONDS)
    except BadSignature:
        return None
