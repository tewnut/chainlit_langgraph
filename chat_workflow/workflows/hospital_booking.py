from ..llm import llm_factory, ModelCapability
from ..tools import BasicToolNode
from chainlit.input_widget import Select
from chat_workflow.workflows.base import BaseWorkflow, BaseState
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, END
from typing import List, Literal
from typing_extensions import TypedDict
import chainlit as cl

class BookingInfo(TypedDict):
    """Structured response for hospital booking."""
    chief_complaint: str  # Brief chief complaint
    date: str  # Preferred date or time
    location: str  # One of the 3 predefined locations

class Router(TypedDict):
    """Worker to route to next. If no workers are needed, route to __end__"""
    next: Literal["chief_complaint_consultant", "date_picker", "location_selector", "responder"]
    messages: str

class ChiefComplaintResponse(TypedDict):
    """Structured response for chief_complaint_node."""
    chief_complaint: str  # Chief complaint provided by the user
    messages: str  # Any follow-up messages or notes

class DatePickerResponse(TypedDict):
    """Structured response for date_picker_node."""
    date: str  # Preferred date or time
    messages: str  # Any follow-up messages or notes

class LocationSelectorResponse(TypedDict):
    """Structured response for location_selector_node."""
    location: str  # Selected location based on the user's input
    messages: str  # Any follow-up messages or notes

# Define the State Class
class GraphState(BaseState):
    chat_model: str  # Model used by the chatbot
    chief_complaint: str  # Chief complaint
    date: str  # Preferred date or time
    location: str  # Selected location
    next: str

# Define the Workflow
class HospitalBookingWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        self.capabilities = {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING}
        self.tools = []

    @classmethod
    def name(self) -> str:
        return "Hospital Booking Assistant"

    @property
    def output_chat_model(self) -> str:
        return "gpt-4o-mini"  # Example: Use GPT-4 for final responses

    @classmethod
    def chat_profile(cls):
        return cl.ChatProfile(
            name=cls.name(),
            markdown_description="An assistant that helps with hospital bookings by selecting the appropriate location and scheduling based on the user's needs.",
            icon="https://cdn3.iconfinder.com/data/icons/hospital-outline/128/hospital-bed.png",
        )

    @property
    def chat_settings(self) -> cl.ChatSettings:
        return cl.ChatSettings([
            Select(
                id="chat_model",
                label="Chat Model",
                values=sorted(llm_factory.list_models(
                    capabilities=self.capabilities)),
                initial_index=-1,
            )
        ])

    def create_default_state(self) -> GraphState:
        # Initialize the default state variables
        return {
            "name": self.name(),
            "messages": [],
            "chat_model": "gpt-4o-mini",
            "chief_complaint": "",
            "date": "",
            "location": "",
            "next": "chief_complaint",
        }

    def create_graph(self) -> StateGraph:
        # Create the state graph
        graph = StateGraph(GraphState)

        # Add nodes (agents)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("chief_complaint_consultant", self.chief_complaint_node)
        graph.add_node("date_picker", self.date_picker_node)
        graph.add_node("location_selector", self.location_selector_node)
        graph.add_node("responder", self.responder_node)

        # Add edges (transitions)
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", lambda state: state["next"])
        graph.add_edge("chief_complaint_consultant", "supervisor")
        graph.add_edge("date_picker", "supervisor")
        graph.add_edge("location_selector", "supervisor")
        graph.add_edge("responder", END)

        return graph

    ### Define Node Methods ###
    async def supervisor_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: supervisor_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a supervisor coordinating tasks between agents to collect hospital booking details.  
Based on the current state:
- Assign the next agent (`chief_complaint_consultant`, `date_picker`, `location_selector`, or `responder`) to perform a task.
- Include any instructions or context for the agent in your response.
- If the booking is complete, assign the responder agent to communicate with the user.

Current progress:
- Chief complaint: {chief_complaint}
- Preferred date: {date}
- Selected location: {location}
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(Router)

        chain: Runnable = prompt | llm
        response = await chain.ainvoke(state, config=config)
        print(f"supervisor_node response: {response}")
        state.update({"messages": [{"role": "ai", "content": response["messages"]}], "next": response["next"]})
        return state

    async def chief_complaint_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: chief_complaint_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an agent responsible for collecting the user's chief complaint.  
Ask the user for a brief description of their health concern or reason for booking.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(ChiefComplaintResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"chief_complaint_node response: {response}")

        state.update({
            "chief_complaint": response["chief_complaint"],
            "messages": [{"role": "assistant", "content": response["messages"]}]
        })
        return state

    async def date_picker_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: date_picker_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an agent responsible for selecting the user's preferred date or time for the appointment.  
Ask the user for their preferred schedule.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(DatePickerResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"date_picker_node response: {response}")

        state.update({
            "date": response["date"],
            "messages": [{"role": "assistant", "content": response["messages"]}]
        })
        return state

    async def location_selector_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: location_selector_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an agent responsible for selecting the most appropriate hospital location based on the user's chief complaint.  
Available locations:
- BV Chợ Rẫy: General health and emergency services.
- BV ĐH Y Dược: Specialized in internal medicine and diagnostics.
- BH Ung Bướu: Focused on cancer treatment.

Chief complaint: {chief_complaint}

Choose the most suitable location and explain the reasoning.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(LocationSelectorResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"location_selector_node response: {response}")

        state.update({
            "location": response["location"],
            "messages": [{"role": "assistant", "content": response["messages"]}]
        })
        return state

    async def responder_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: responder_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful AI assistant confirming the hospital booking details to the user.
Booking details:
- Chief complaint: {chief_complaint}
- Preferred date: {date}
- Selected location: {location}

Confirm the details with the user or ask for any corrections if necessary.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        )
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"responder_node response: {response}")
        state.update({"messages": [{"role": "assistant", "content": response}]})
        return state
