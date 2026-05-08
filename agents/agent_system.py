from typing import Any, Optional
from langchain_core.messages import AnyMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from utils.models import get_llm
from .tool_factory import BaseToolFactory
from .answer_critic import AnswerCritic, Critique
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
import json

class AgentResult(BaseModel):
    iteration: int
    question: str
    answer: str
    context: list[str]
    critique: Critique

class AgentSystem:
    def __init__(self):
        self.model = get_llm(0.0)
        self.critic = AnswerCritic()
        self.graph = None
        self.tool_manager: Optional[BaseToolFactory] = None
        self.system_prompt = ""

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
        self.tool_manager = tool_manager
        self.system_prompt = system_prompt

        tools = self.tool_manager.get_tools()        
        model_with_tools = self.model.bind_tools(tools)

        def call_model(state: MessagesState):
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            response = model_with_tools.invoke(messages)

            # Si hay mas de una tool nos quedamos con la ultima, para evitar la ejecución en paralelo
            if response.tool_calls and len(response.tool_calls) > 1:
                response.tool_calls = response.tool_calls[:1]

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

        config = {
            "configurable": {"thread_id": thread_id},
        }

        messages: list[AnyMessage] = []

        for chunk  in self.graph.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="updates",
        ):            
            for _, update in chunk.items():
                if "messages" in update:
                    for msg in update["messages"]:
                        self._render_completed_message(msg)
                        messages.append(msg)

        answer = messages[-1].content
        return str(answer), messages

    def _extract_context(self, messages: list[AnyMessage]) -> list:
        tools_msgs = [
            msg for msg in messages if isinstance(msg, ToolMessage)
        ]

        if not tools_msgs:
            return []

        content = tools_msgs[-1].content

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
        if not self.tool_manager:
            return False

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and self.tool_manager.is_terminal_tool(msg.name):
                return True
        return False

    def answer(self, question: str, thread_id: str = "default", max_iterations: int = 2) -> dict[str, Any]:
        iterations: list[AgentResult] = []
        current_question = question

        for i in range(max_iterations):
            self._log(f"ITERATION {i + 1}")
            self._log("QUESTION", current_question, indent=2)

            answer, messages = self._run_agent(current_question, thread_id)

            if self._is_terminal(messages):
                iterations.append(AgentResult(
                    iteration=i+1,
                    question=current_question,
                    answer=answer,
                    context=[],
                    critique=Critique(
                        is_complete=True,
                        is_faithful=True,
                        missing_info=[]
                    )
                ))
                break

            context = self._extract_context(messages)
            self._log("CONTEXT", context, indent=2)

            if not context:
                iterations.append(AgentResult(
                    iteration=i+1,
                    question=current_question,
                    answer="This information is not in the knowledge base.",
                    context=[],
                    critique=Critique(
                        is_complete=False,
                        is_faithful=True,
                        missing_info=[]
                    )
                ))
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

            iterations.append(AgentResult(
                iteration=i+1,
                question=current_question,
                answer=answer,
                context=context,
                critique=critique
            ))

            if critique.is_complete and critique.is_faithful:
                break

            if critique.missing_info and i < max_iterations - 1:
                current_question = (
                    f"{current_question}\n"
                    f"Follow-up requirements: {' '.join(critique.missing_info)}"
                )

        final = iterations[-1]

        self._log("LOOP FINISHED")

        return {
            "question": question,
            "answer": final.answer,
            "iterations": iterations,
            "final_critique": final.critique,
        }    
