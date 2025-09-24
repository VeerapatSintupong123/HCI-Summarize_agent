from smolagents import OpenAIServerModel, tool, ToolCallingAgent
from datetime import datetime
from dotenv import load_dotenv
import os

# --- Load environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Fake backend functions ---
def get_weather_report_at_coordinates(coordinates, date_time):
    return [28.0, 0.35, 0.85]  # temp °C, rain risk, wave height

def convert_location_to_coordinates(location):
    return [3.3, -42.0]  # fake lon, lat

# --- Tool for weather worker ---
@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place you want the weather for.
        date_time: the date and time in format '%m/%d/%y %H:%M:%S'.
    """

    lon, lat = convert_location_to_coordinates(location)
    try:
        parsed_date = datetime.strptime(date_time, "%m/%d/%y %H:%M:%S")
    except Exception as e:
        raise ValueError(
            "Conversion of `date_time` to datetime format failed, "
            "must be '%m/%d/%y %H:%M:%S'. Full trace: " + str(e)
        )
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates(
        (lon, lat), parsed_date
    )
    return (
        f"Weather report for {location} at {parsed_date}: "
        f"Temperature {temperature_celsius}°C, "
        f"Rain risk {risk_of_rain*100:.0f}%, "
        f"Wave height {wave_height}m."
    )

# --- Model ---
model = OpenAIServerModel(
    model_id="gemini-2.5-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

# --- Worker Agent (weather tool) ---
weather_worker = ToolCallingAgent(
    model=model,
    tools=[get_weather_api],
    name="Weather_Worker",
    description="Handles weather queries with get_weather_api.",
    stream_outputs=False,
)

# --- Orchestrator Agent (delegates to workers) ---
orchestrator = ToolCallingAgent(
    model=model,
    tools=[], 
    managed_agents=[weather_worker],  # worker agent is managed
    name="Orchestrator",
    description="Routes tasks to the right worker.",
    stream_outputs=False
)

# --- Run Orchestrator ---
if __name__ == "__main__":
    query = "What will the weather be in Bangkok, Thailand on 09/23/25 15:00:00?"
    result = orchestrator.run(query)
    print("\nFinal Answer:\n", result)