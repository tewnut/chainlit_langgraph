"""
Simple demo of integration with ChainLit and LangGraph.
"""
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from chat_workflow.module_discovery import discover_modules

discovered_workflows = discover_modules()


@cl.set_chat_profiles
async def chat_profile():
    profiles = []
    for workflow in discovered_workflows.values():
        profiles.append(workflow.chat_profile)
    return profiles


@cl.on_chat_start
async def on_chat_start():
    workflow_name = "simple_chat"  # Default workflow
    workflow = discovered_workflows[workflow_name]

    graph = workflow.create_graph()
    state = workflow.create_default_state()

    cl.user_session.set("graph", graph.compile())
    cl.user_session.set("state", state)
    cl.user_session.set("current_workflow", workflow_name)

    await update_state_by_settings(await workflow.get_chat_settings())


@cl.on_settings_update
async def update_state_by_settings(settings: cl.ChatSettings):
    state = cl.user_session.get("state")
    for key in settings.keys():
        if key not in state:
            print(f"Setting {key} not found in state")
            continue
        print(f"Setting {key} to {settings[key]}")
        state[key] = settings[key]
    cl.user_session.set("state", state)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the graph and state from the user session
    graph: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")
    workflow = discovered_workflows[cl.user_session.get("current_workflow")]

    # Append the new message to the state
    state["messages"] += [HumanMessage(content=message.content)]

    # Stream the response to the UI
    ui_message = None
    total_content: str = ""
    async for event in graph.astream_events(state, version="v1"):
        string_content = ""
        if event["event"] == "on_chat_model_stream" and event["name"] == workflow.output_chat_model:
            content = event["data"]["chunk"].content or ""
            if type(content) == str:
                string_content += content
            elif type(content) == list and len(content) > 0:
                if type(content[0]) == str:
                    string_content += " ".join(content)
                elif type(content[0]) == dict and "text" in content[0]:
                    string_content += " ".join([c["text"] for c in content])
            else:
                string_content = ""
            total_content += string_content
            if ui_message is None:
                ui_message = cl.Message(content=string_content)
                await ui_message.send()
            else:
                await ui_message.stream_token(token=string_content)
    await ui_message.update()

    # Update State
    state["messages"] += [AIMessage(content=total_content)]
    cl.user_session.set("state", state)
