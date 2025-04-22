import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from typing_extensions import TypedDict

from agents import Agent, Runner, RunContextWrapper, function_tool, RunResult, FunctionToolResult, ToolsToFinalOutputResult

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Define the input type for fetch_weather
class Location(TypedDict):
    lat: float
    long: float

@function_tool
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    print("--- Tool 'fetch_weather' called with location: %s ---" % location)
    # In real life, we'd fetch the weather from a weather API
    return "sunny"

@function_tool(name_override="fetch_data")
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    print("--- Tool 'read_file' (fetch_data) called with path: %s, dir: %s ---" % (path, directory))
    # In real life, we'd read the file from the file system
    return "<file contents>"

# Custom function to replicate StopAtTools(["fetch_weather"])
async def stop_if_fetch_weather_called(
    context: RunContextWrapper[Any],
    results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Checks if 'fetch_weather' was called. If yes, stops and returns its output."""
    for result in results:
        # Check the tool name via result.tool.name
        if hasattr(result, 'tool') and hasattr(result.tool, 'name') and result.tool.name == "fetch_weather":
            return ToolsToFinalOutputResult(is_final_output=True, final_output=result.output)

    return ToolsToFinalOutputResult(is_final_output=False)


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with OPENAI_API_KEY='your-key-here'")
        return

    # Agent uses the OPENAI_API_KEY environment variable automatically

    # Instantiate the agent, specifying the tool to stop at
    agent = Agent(
        model="gpt-4o", # or your preferred model
        name="WeatherBot",
        instructions="You are a helpful assistant.",
        tools=[fetch_weather, read_file], 
        tool_use_behavior={"stop_at_tool_names": ["fetch_weather"]}
    )

    print(f"Agent created with tool_use_behavior={agent.tool_use_behavior}")

    # Message designed to trigger the fetch_weather tool
    message = "What's the weather like at latitude 34.05 and longitude -118.24?"
    print("\nRunning agent with message: '%s'" % message)

    # Use a Runner to execute the agent
    runner = Runner()
    result: RunResult = await runner.run(agent, input=message)

    print("\nAgent run finished.")
    print(f"Final Output: {result.final_output}")

    # --- Verification ---
    # Check if the final output is exactly the output of the 'fetch_weather' tool
    expected_output = "sunny"
    assert result.final_output == expected_output, f"Expected final output '{expected_output}', but got '{result.final_output}'"
    print("\nAssertion Passed: Final output matches the 'fetch_weather' tool result as expected.")


if __name__ == "__main__":
    asyncio.run(main())
