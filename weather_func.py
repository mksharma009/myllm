"""Declared all fucntions related to weather tool here."""

import logging
import requests
from typing import Tuple, Optional
from geopy.geocoders import Nominatim

logging = logging.getLogger(__name__)

GEOLOCATOR_USER_AGENT = "gemini_weather_app"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
def get_lat_long(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude and longitude for a given location using OpenStreetMap Nominatim API.

    Args:
        location: The location name to geocode.

    Returns:
        A tuple of (latitude, longitude) or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent=GEOLOCATOR_USER_AGENT)
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        logging.error(f"Error geocoding location '{location}': {e}")
    return None, None


def get_weather(location: str) -> str:
    """Get weather information for a given location.

    Args:
        location: The location name to get weather for.

    Returns:
        A string with weather information or an error message.
    """
    lat, lon = get_lat_long(location)
    if lat is None or lon is None:
        return f"Sorry, I couldn't find the location: {location}."

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current_weather = data.get("current_weather", {})
        temperature = current_weather.get("temperature")
        weather_code = current_weather.get("weathercode")

        if temperature is not None:
            return f"The current temperature in {location} is {temperature}°C with weather code {weather_code}."
        else:
            return f"Sorry, I couldn't retrieve weather data for {location}."

    except requests.RequestException as e:
        logging.error(f"Error fetching weather data for '{location}': {e}")
        return f"Sorry, I couldn't fetch the weather information for {location} at this time."