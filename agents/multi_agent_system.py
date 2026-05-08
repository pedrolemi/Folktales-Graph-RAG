from .agentic_rag import AgenticRAG
from neo4j_manager import Neo4jManager
from .terminal_tool_factory import TerminalToolFactory
from .agent_system import AgentSystem

class MultiAgentSystem(AgentSystem):
    def __init__(self, neo4j_manager: Neo4jManager):
        super().__init__()

        self.event_rag = AgenticRAG(
            neo4j_manager,
            embedding_index="event_embeddings",
            fulltext_index="event_fulltext",
            node_label="Event",
            text_property="description",
            tool_name="event_tool"
        )

        self.character_rag = AgenticRAG(
            neo4j_manager,
            embedding_index="character_embeddings",
            fulltext_index="character_fulltext",
            node_label="Character",
            text_property="description",
            tool_name="character_tool"
        )

        self.place_rag = AgenticRAG(
            neo4j_manager,
            embedding_index="place_embeddings",
            fulltext_index="place_fulltext",
            node_label="Place",
            text_property="description",
            tool_name="place_tool"
        )

        self.object_rag = AgenticRAG(
            neo4j_manager,
            embedding_index="object_embeddings",
            fulltext_index="object_fulltext",
            node_label="Object",
            text_property="description",
            tool_name="object_tool"
        )

        tool_manager = TerminalToolFactory()

        tool_manager.register_custom_tool(self.event_rag.as_tool(max_iterations=2))
        tool_manager.register_custom_tool(self.character_rag.as_tool(max_iterations=2))
        tool_manager.register_custom_tool(self.place_rag.as_tool(max_iterations=2))
        tool_manager.register_custom_tool(self.object_rag.as_tool(max_iterations=2))

        system_prompt="""
You are a multi-agent coordinator for a folktale analysis system.

Your task is to direct each user request to ONE appropriate tool. You do NOT answer directly unless a tool is invoked.

AVAILABLE TOOLS:

1. greeting
- Use for greetings, farewalls, thanks or casual conversation.

2. skills
- Use when the user asks about system capabilities.

3. out_of_scope
- Use when the question is NOT related to stories, narratives or story analysis.

4. even_tool
- Use for quequeestions about EVENTS in a story.

An Event is:
- a story action or occurrence.
- a plot step or scene.
- a narrative transition.
- a causal moment in the story.
- a step in story progression.

Use event_tool for:
- what happens in the story.
- plot progression.
- story structure.
- sequence of events.
- cause and effect in the narrative.
- turning points in the story.

5. character_tool (CHARACTER GRAPH TOOL)
- Use for questions about CHARACTERS in the story graph.

A Character is:
- an entity in the story that plays a role within it (person or creature).
- connected to events and other characters in the graph.

Use character_tool for:
- character descriptions.
- character relationships.
- character interactions.
- character roles in the story.
- character development.
- who did what (when focus is on the person, not the event).

6. place_tool
Use for PLACES in the story, where an event occurs or a character lives.

A Place is:
- a physical or conceptual location in the story graph.
- where events happen.

Use place_tool for:
- story settings.
- locations of events.
- where the characters or events take place.
- where a character lives.

7. object_tool
Use for OBJECTS in a story.

An Object is:
- an inanimate entity in the world of history.
- something that characters can use, carry or interact with.
- NOT a character and NOT a place.

Examples of objects:
- weapons (swords, guns, tools).
- artifacts (rings, crowns, magical items)
- documents (letters, books, maps)
- consumables (potions, goods)
- keys, relics, treasures

Use object_tool for:
- what objects exist in the story.
- who possesses or uses an object.
- how objects are used in events.
- object significance in the narrative.
- object relationships to characters or events.

ROUTING RULES:

You must select EXACTLY ONE tool.

Determine PRIMARY focus:
EVENT FOCUS -> event_tool
- what happens.
- actions, sequences, plot progression.

CHARACTER FOCUS -> character_tool
- who is involved.
- identity, agency, relationships.

PLACE FOCUS -> place_tool
- where something happens.
- settings and locations.

OBJECT FOCUS -> object_tool
- what physical items exist.
- what characters use, carry or interact with.

If multiple categories appear, choose the PRIMARY intent. If it is not clear:
- prefer event_tool for narrative questions.
- prefer character_tool if "who" is dominant.
- prefer place_tool if "where" is dominant.
- prefer object_tool if "what item" is dominant.
"""
        
        self._build_agent(tool_manager, system_prompt)

    