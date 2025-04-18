from typing import Any, Dict, List

from agents import (  # Assuming 'agents' is the library/module name
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
)


def process_agent_step_outputs(agent_run) -> List[Dict[str, Any]]:
    """Processes the output items from an agent run into a list of dictionaries."""
    processed_outputs = []
    for new_item in agent_run.new_items:
        agent_name = getattr(getattr(new_item, "agent", None), "name", "UnknownAgent")
        output_dict = {"agent": agent_name, "type": new_item.__class__.__name__}

        if isinstance(new_item, MessageOutputItem):
            output_dict["message"] = ItemHelpers.text_message_output(new_item)
        elif isinstance(new_item, HandoffOutputItem):
            source_name = getattr(
                getattr(new_item, "source_agent", None), "name", "UnknownSource"
            )
            target_name = getattr(
                getattr(new_item, "target_agent", None), "name", "UnknownTarget"
            )
            output_dict["message"] = f"Handed off from {source_name} to {target_name}"
            output_dict["source_agent"] = source_name
            output_dict["target_agent"] = target_name
        elif isinstance(new_item, ToolCallItem):
            tool_name = getattr(
                new_item.raw_item,
                "name",
                getattr(new_item.raw_item, "type", "<tool name unknown>"),
            )
            output_dict["message"] = f"Called the {tool_name} tool"
            output_dict["tool_name"] = tool_name
            # Potentially add tool input arguments here if needed/available
        elif isinstance(new_item, ToolCallOutputItem):
            tool_output = getattr(new_item, "output", "N/A")
            output_dict["message"] = f"Tool call output: {tool_output}"
            output_dict["tool_output"] = tool_output
        else:
            output_dict["message"] = f"Skipping item: {new_item.__class__.__name__}"

        processed_outputs.append(output_dict)

    return processed_outputs
