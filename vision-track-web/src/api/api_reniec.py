# filepath: vision-track-web/src/api/api_reniec.py
from typing import List, Optional
import logging
import requests

class ApisNetPe:
    def __init__(self, token: str = None) -> None:
        self._api_token = token
        self._api_url = "https://api.apis.net.pe"

    def _get(self, path: str, params: dict):
        url = f"{self._api_url}{path}"

        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Referer": "python-requests"
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            logging.warning(f"{response.url} - invalida params", params=params)
            logging.warning(response.text)
        elif response.status_code == 403:
            logging.warning(f"{response.url} - IP blocked")
        elif response.status_code == 429:
            logging.warning(f"{response.url} - Many requests add delay")
        elif response.status_code == 401:
            logging.warning(f"{response.url} - Invalid token or limited")
        else:
            logging.warning(f"{response.url} - Server Error status_code={response.status_code}")
        return None

    def get_person(self, dni: str) -> Optional[dict]:
        return self._get("/v2/reniec/dni", {"numero": dni})

    def get_company(self, ruc: str) -> Optional[dict]:
        return self._get("/v2/sunat/ruc", {"numero": ruc})

    def get_exchange_rate(self, date: str) -> dict:
        return self._get("/v2/sunat/tipo-cambio", {"fecha": date})

    def get_exchange_rate_today(self) -> dict:
        return self._get("/v2/sunat/tipo-cambio", {})

    def get_exchange_rate_for_month(self, month: int, year: int) -> List[dict]:
        return self._get("/v2/sunat/tipo-cambio", {"month": month, "year": year})