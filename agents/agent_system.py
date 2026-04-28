from typing import Any
from langchain_core.messages import AnyMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from utils.models import get_llm
from .tool_factory import BaseToolFactory
from .answer_critic import AnswerCritic
from langchain.agents.middleware import AgentMiddleware
from langgraph.prebuilt import ToolNode
import json

# class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
#     def wrap_model_call(self, request, handler):
#         request.model_settings.pop("parallel_tool_calls", None)
#         return handler(request)

#     async def awrap_model_call(self, request, handler):
#         request.model_settings.pop("parallel_tool_calls", None)
#         return await handler(request)

class AgentSystem:
    def __init__(self):
        self.model = get_llm(0.0)
        self.critic = AnswerCritic()
        self.graph = None
        self.tool_manager = None
        self.system_prompt = None

    # def _render_message_chunk(self, token: AIMessageChunk):
    #     print("\n")
    #     if token.text:
    #         print(token.text, end="|")
    #     if token.tool_call_chunks:
    #         print(token.tool_call_chunks)

    def _log(self, title: str, content: Any = "", indent: int = 0):
        prefix = " " * indent
        print(f"\n{prefix}{'=' * 10} {title} {'=' * 10}")

        if content:
            if isinstance(content, (dict, list)):
                print(prefix + json.dumps(content, indent=2, ensure_ascii=False))
            else:
                print(prefix + str(content))

    def _render_completed_message(self, message: AnyMessage):
        print("\n")
        if isinstance(message, AIMessage):
            if message.tool_calls:
                self._log(
                    "AI -> TOOL CALL",
                    message.tool_calls,
                    indent=2,
                )
            else:
                self._log(
                    "AI -> FINAL ANSWER",
                    message.content,
                    indent=2,
                )
        elif isinstance(message, ToolMessage):
            self._log(
                f"TOOL -> RESPONSE ({message.name})",
                message.content_blocks,
                indent=4,
            )

    def _build_agent(self, tool_manager: BaseToolFactory, system_prompt: str):
        # self.tool_manager = tool_manager
        # tools = self.tool_manager.get_tools()

        # self.agent = create_agent(
        #     model=self.model,
        #     tools=tools,
        #     system_prompt=system_prompt,
        #     checkpointer=InMemorySaver(),
        #     middleware=[DisableParallelToolCallsMiddleware()]
        # )

        self.tool_manager = tool_manager
        self.system_prompt = system_prompt
        tools = self.tool_manager.get_tools()

        # Create a mapping from tool name to callable tool
        # tool_map: dict[str, BaseTool] = {tool.name: tool for tool in tools}

        # Bind tools to the model
        model_with_tools = self.model.bind_tools(tools)

        def call_model(state: MessagesState):
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            response = model_with_tools.invoke(messages)

            # if response.tool_calls and len(response.tool_calls) > 1:
                # response.tool_calls = response.tool_calls[:1]
            return {"messages": [response]}
        
        tool_node = ToolNode(tools)
        
        def should_continue(state: MessagesState):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile(checkpointer=InMemorySaver())

    
    def _run_agent(self, question: str, thread_id: str):
        input_message = {
            "role": "user",
            "content": question
        }

        # config = {
            # "configurable": {"thread_id": thread_id},
        # }

        all_messages: list[AnyMessage] = []

        for chunk  in self.graph.stream(
            {"messages": [input_message]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            # print(chunk)
            
            # if chunk["type"] == "messages":
            #     token, _ = chunk["data"]
            #     if isinstance(token, AIMessageChunk):
            #         self._render_message_chunk(token)
            for _, update in chunk.items():
                if "messages" in update:
                    for msg in update["messages"]:
                        self._render_completed_message(msg)
                        all_messages.append(msg)

        answer = all_messages[-1].content
        return str(answer), all_messages

        # result = self.agent.invoke(
            # {"messages": [input_message]},
            # config=config,
        # )

        # messages: list[AnyMessage] = result["messages"]

        # for msg in messages:
            # msg.pretty_print()

        # return str(messages[-1].content), messages

    def _extract_context(self, messages: list[AnyMessage]) -> list:
        intermediate_steps = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]

        if not intermediate_steps:
            return []

        last_tool_msg = intermediate_steps[-1]
        content = last_tool_msg.content

        try:
            if isinstance(content, str):
                content = json.loads(content)
        except json.JSONDecodeError:
            return [str(content)]

        if isinstance(content, dict):
            return content.get("context", [])
        elif isinstance(content, list):
            return list(map(str, content))
        
        return [str(content)]
    
    def _is_terminal(self, messages: list[AnyMessage]) -> bool:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if self.tool_manager.is_terminal_tool(msg.name):
                    return True
        return False

    def answer(self, question: str, thread_id: str = "default", max_iterations: int = 2) -> dict[str, Any]:
        iterations = []
        current_question = question

        for i in range(max_iterations):
            self._log(f"ITERATION {i + 1}")
            self._log("QUESTION", current_question, indent=2)

            answer, messages = self._run_agent(current_question, thread_id)

            if self._is_terminal(messages):
                iterations.append({
                    "iteration": i + 1,
                    "question": current_question,
                    "answer": answer,
                    "context": [],
                    "critique": {"is_complete": True, "is_faithful": True, "missing_info": []}
                })
                break

            context = self._extract_context(messages)

            self._log("CONTEXT", context, indent=2)

            if not context:
                answer = "This information is not in the knowledge base."

                iterations.append({
                    "iteration": i + 1,
                    "question": current_question,
                    "answer": answer,
                    "context": [],
                    "critique": {"is_complete": False, "is_faithful": True, "missing_info": []},
                })

                break

            critique = self.critic.critique(
                current_question,
                context,
                answer
            )

            self._log("CRITIQUE", {
                "is_complete": critique.is_complete,
                "is_faithful": critique.is_faithful,
                "missing_info": critique.missing_info,
            }, indent=2)

            iterations.append({
                "iteration": i + 1,
                "question": current_question,
                "answer": answer,
                "context": context,
                "critique": critique,
            })

            # Stop condition
            if critique.is_complete and critique.is_faithful:
                break

            # Refinement
            if critique.missing_info and i < max_iterations - 1:
                current_question = (
                    f"{current_question}\n"
                    f"Additional questions: {' '.join(critique.missing_info)}"
                )

        final = iterations[-1]

        self._log("LOOP FINISHED")

        return {
            "question": question,
            "answer": final["answer"],
            "iterations": iterations,
            "final_critique": final["critique"],
        }
    
