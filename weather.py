import requests

LAT = 32.7948
LON = -116.9625
TIMEZONE = "America/Los_Angeles"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUEST_TIMEOUT = 10


def _c_to_f(c: float | None) -> float | None:
    return None if c is None else round(c * 9 / 5 + 32, 1)


def _mm_to_in(mm: float | None) -> float | None:
    return None if mm is None else round(mm / 25.4, 2)


def _parse_daily(payload: dict) -> list[dict]:
    daily = payload["daily"]
    return [
        {
            "date": d,
            "temp_max_f": _c_to_f(tmax),
            "temp_min_f": _c_to_f(tmin),
            "precip_in": _mm_to_in(precip),
        }
        for d, tmax, tmin, precip in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
        )
    ]


def get_forecast(days: int = 7) -> list[dict]:
    """Daily forecast for the next N days at El Cajon, CA from Open-Meteo."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": DAILY_VARS,
        "timezone": TIMEZONE,
        "forecast_days": days,
    }
    response = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return _parse_daily(response.json())


def get_historical(start_date: str, end_date: str) -> list[dict]:
    """Daily historical weather between two YYYY-MM-DD dates at El Cajon, CA."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": DAILY_VARS,
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return _parse_daily(response.json())


def format_weather(daily_data: list[dict]) -> str:
    """Pretty-print daily weather as a fixed-width plain-text table."""
    if not daily_data:
        return "(no weather data)"
    header = f"{'Date':<12} {'Max (°F)':>9} {'Min (°F)':>9} {'Precip (in)':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for row in daily_data:
        tmax = f"{row['temp_max_f']:.1f}" if row["temp_max_f"] is not None else "N/A"
        tmin = f"{row['temp_min_f']:.1f}" if row["temp_min_f"] is not None else "N/A"
        precip = f"{row['precip_in']:.2f}" if row["precip_in"] is not None else "N/A"
        lines.append(f"{row['date']:<12} {tmax:>9} {tmin:>9} {precip:>12}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Forecast (3 days) ===")
    print(format_weather(get_forecast(days=3)))

    print("\n=== Historical (last 5 days of 2025) ===")
    print(format_weather(get_historical("2025-12-27", "2025-12-31")))