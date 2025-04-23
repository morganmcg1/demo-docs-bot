import ast
import json

def map_outputs_to_agents(outputs, initial_agent_name):
    """
    Walks the output list, updating the current agent only AFTER the function_call_output
    of a transfer_to_* function call. Each item is mapped to the agent that produced it.
    Returns a dict with agents as keys and their respective outputs as values.
    """
    segmented_history = {}
    current_agent_name = initial_agent_name  #
    pending_handoff_call_id = None
    
    for item in outputs:
        # Assign the current agent to this item BEFORE any possible update
        if current_agent_name not in segmented_history:
            segmented_history[current_agent_name] = []
        
        segmented_history[current_agent_name].append(item)

        # Check for a handoff function call
        if item.get("type") == "function_call" and item.get("name", "").startswith("transfer_to_"):
            pending_handoff_call_id = item.get("call_id")
            # Don't update agent yet; wait for the output

        # Check for the function_call_output that matches the handoff
        elif (
            item.get("type") == "function_call_output"
            and pending_handoff_call_id
            and item.get("call_id") == pending_handoff_call_id
        ):
            try:
                # First try to get output directly
                output = item.get("output")
                
                # Parse output if it's a string
                if isinstance(output, str):
                    try:
                        # Try to parse as JSON first
                        parsed_output = json.loads(output)
                    except json.JSONDecodeError:
                        # If that fails, try to parse as Python literal
                        parsed_output = ast.literal_eval(output)
                    
                    next_agent = parsed_output.get("assistant")
                elif isinstance(output, dict):
                    next_agent = output.get("assistant")
                else:
                    next_agent = None
                
                # Update the current agent if we found a next agent
                if next_agent:
                    # Actually update the current_agent_name here
                    current_agent_name = next_agent
                    # Initialize the new agent's history if needed
                    if current_agent_name not in segmented_history:
                        segmented_history[current_agent_name] = []
                
            except Exception as e:
                print(f"Error extracting next agent: {e}")
                # Keep the current agent if there's an error
            
            # Reset the pending handoff flag
            pending_handoff_call_id = None

    return segmented_history