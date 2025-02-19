import base64


def url_save_str(s: str) -> str:
    return s.replace("/", "_")


def url_save_hash(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()
