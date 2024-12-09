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
    hospital: str  # One of the 3 predefined hospitals

class Router(TypedDict):
    """Worker to route to next. If no workers are needed, route to __end__"""
    next: Literal["hospital_finder", "date_picker", "responder"]
    messages: str

class GeneralPractionerResponse(TypedDict):
    """Structured response for hospital_finder_node."""
    chief_complaint: str  # Chief complaint provided by the user
    hospital: str  # Selected hospital based on the user's input
    messages: str  # Any follow-up messages or notes

class DatePickerResponse(TypedDict):
    """Structured response for date_picker_node."""
    date: str  # Preferred date or time
    messages: str  # Any follow-up messages or notes

# Define the State Class
class GraphState(BaseState):
    chat_model: str  # Model used by the chatbot
    chief_complaint: str  # Chief complaint
    date: str  # Preferred date or time
    hospital: str  # Selected hospital
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
            markdown_description="An assistant that helps with hospital bookings by selecting the appropriate hospital and scheduling based on the user's needs.",
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
            "hospital": "",
            "next": "responder",
        }

    def create_graph(self) -> StateGraph:
        # Create the state graph
        graph = StateGraph(GraphState)

        # Add nodes (agents)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("hospital_finder", self.hospital_finder_node)
        graph.add_node("date_picker", self.date_picker_node)
        graph.add_node("responder", self.responder_node)

        # Add edges (transitions)
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", lambda state: state["next"])
        graph.add_edge("hospital_finder", "supervisor")
        graph.add_edge("date_picker", "supervisor")
        graph.add_edge("responder", END)

        return graph

    ### Define Node Methods ###
    async def supervisor_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: supervisor_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a supervisor coordinating tasks between the AI agents (`hospital_finder`, `date_picker`, `responder`) to help user book for a hospital visit.  

Instructions
- If user input is required, assign the 'responder'.
- If chief_complaint or hospital not clear, assign the 'hospital_finder'.
- If date not clear, assign the 'date_picker'.
- When done, assign the 'responder'.

Current progress:
- chief_complaint: {chief_complaint}
- date: {date}
- hospital: {hospital}
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        print(f"supervisor_node state: {state}")

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(Router)

        chain: Runnable = prompt | llm
        response = await chain.ainvoke(state, config=config)
        print(f"supervisor_node response: {response}")
        state.update({"messages": [{"role": "ai", "content": response["messages"]}], "next": response["next"]})
        return state

    async def hospital_finder_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        
        print("Running: hospital_finder_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a hospital finder, supervised by an AI Agent. You are responsible for clarifying user's chief complaint in order to suggest the right hospital.

Available hospitals for booking:
- BV Chợ Rẫy: General health and emergency services.
- BV ĐH Y Dược: Specialized in internal medicine and diagnostics.
- BV Ung Bướu: Focused on cancer treatment.

Instructions:
- Update chief_comlaint and hospital.
- Include a brief status message to tell the supervisor if you are done, or you need additional clarifications from user.

Current progress:
- chief_comlaint: {chief_complaint}
- hospital: {hospital}

        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(GeneralPractionerResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"hospital_finder_node response: {response}")

        state.update({
            "chief_complaint": response["chief_complaint"],
            "hospital": response["hospital"],
            "messages": [{"role": "assistant", "content": response["messages"]}]
        })
        return state

    async def date_picker_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: date_picker_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an scheduling agent, supervised by an AI Agent.

Instructions:
- Update the booking date based on user preference.
- If user hasn't indicate their time prefernece, ask the supervisor to provide it.
- Booking date must fit with user preference and at least 4 hours from now (2024-12-12T13:00:00).

Current bookingdate: {date}
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


    async def responder_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: responder_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful AI Assistant that will converse with the user to help them book a hospital visit. 
Only use the language as user's language.

Current booking details:
- chief_complaint: {chief_complaint}
- date: {date}
- hospital: {hospital}

You are supervised by an AI Agent, which will guide you how to talk with user.

If booking details are all clear, show them to the user when responding.
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
        state.update({"messages": [response]})
        return state
