from utils.regex_utils import snake_case_to_pascal_case
from utils.loader import load_json_folder
from .neo4j_manager import Neo4jManager
from pydantic import BaseModel
from utils.config import get_settings
from typing import Optional
from schemas.folktale import Folktale
from schemas.agent import Agent
from schemas.relationship import Relationship
from schemas.event import Event
from utils.models import get_embeddings
import os


class FolktaleGraphBuilder:
    def __init__(self, neo4j: Neo4jManager, metadata_dir: str = "./metadata"):
        self.neo4j = neo4j
        self.metadata_dir = metadata_dir
        self.structure_dir = os.path.join(metadata_dir, "structure")
        self.collections_dir = os.path.join(metadata_dir, "collections")

        self.settings = get_settings()

        self.model = get_embeddings()

    def clear_database(self):
        self.neo4j.execute_query("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        constraints = [
            # Cuento popular
            "CREATE CONSTRAINT folktale_url IF NOT EXISTS FOR (f:Folktale) REQUIRE f.url IS UNIQUE",

            # IDs
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Character) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT object_id IF NOT EXISTS FOR (o:Object) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT place_id IF NOT EXISTS FOR (p:Place) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",

            # Jerarquías
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT nation_name IF NOT EXISTS FOR (n:Nation) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT trait_name IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT function_name IF NOT EXISTS FOR (f:Function) REQUIRE f.name IS UNIQUE",
        ]
        self.neo4j.create_constraints(constraints)

    def create_vector_databases(self, overwrite: bool = False):
        self.neo4j.create_vector_index(
            index_name="event_embeddings",
            label="Event",
            property_name="embedding",
            dimensions=768,
            similarity="cosine",
            overwrite=overwrite
        )

        self.neo4j.create_vector_index(
            index_name="character_embeddings",
            label="Character",
            property_name="embedding",
            dimensions=768,
            similarity="cosine",
            overwrite=overwrite
        )

        self.neo4j.create_vector_index(
            index_name="place_embeddings",
            label="Place",
            property_name="embedding",
            dimensions=768,
            similarity="cosine",
            overwrite=overwrite
        )

        self.neo4j.create_vector_index(
            index_name="object_embeddings",
            label="Object",
            property_name="embedding",
            dimensions=768,
            similarity="cosine",
            overwrite=overwrite
        )
    
    def load_collections(self):
        collections = load_json_folder(self.collections_dir)

        for label, collection in collections.items():
            label = snake_case_to_pascal_case(label)

            query = f"""
            MERGE (n:{label} {{name: $name}})
            SET n.description = $description
            """

            for name, description in collection.items():
                self.neo4j.execute_query(query, {
                    "name": name,
                    "description": description
                })
    
    def _insert_node(self, label: str, name: str, node: dict, parent: Optional[str] = None):
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n.description = $description
        WITH n
        OPTIONAL MATCH (p:{label} {{name: $parent}})
        FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
            MERGE (p)-[:HAS_CHILD]->(n)
        )
        """

        params = {
            "name": name,
            "description": node.get("description"),
            "parent": parent
        }

        self.neo4j.execute_query(query, params)

        for child_name, child_node in node.get("children", {}).items():
            self._insert_node(label, child_name, child_node, name)

    def load_structures(self):
        structures = load_json_folder(self.structure_dir)

        for node in structures.values():
            label = next(iter(node))
            root_node = node[label]

            self._insert_node(snake_case_to_pascal_case(label), label, root_node)

    def insert_folktale(self, folktale: Folktale):
        query = """
        MERGE (f:Folktale {url: $url})
        SET f.title = $title

        WITH f
        MATCH (g:Genre {name: $genre})
        MERGE (f)-[:HAS_GENRE]->(g)

        WITH f
        MATCH (n:Nation {name: $nation})
        MERGE (f)-[:FROM_NATION]->(n)

        RETURN f
        """

        params = {
            "url": folktale.url,
            "title": folktale.title,
            "genre": folktale.genre,
            "nation": folktale.nation
        }

        self.neo4j.execute_query(query, params)
    
    def insert_entities(self, label: str, items: list[BaseModel], folktale: Folktale):
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n.type = $type,
            n.name = $name,
            n.description = $description,
            n.embedding = $embedding
        """
        
        params_list = [
            item.model_dump(include={"id", "type", "name", "description"})
            for item in items
        ]

        def build_entity_embedding_text(label: str, item: dict, folktale: Folktale) -> str:
            return f"""Folktale: {folktale.title}

{label}: {item["name"]}
Type: {item["type"]}

Description:
{item["description"]}
"""

        descriptions = [
            build_entity_embedding_text(label, item, folktale)
            for item in params_list
        ]

        embeddings = self.model.embed_documents(descriptions)

        for params, embedding in zip(params_list, embeddings):
            params["embedding"] = embedding
            self.neo4j.execute_query(query, params)

    def insert_agents(self, agents: list[Agent], folktale: Folktale):
        agent_query = """
        MERGE (a:Character {id: $agent_id})
        SET a.race = $race,
            a.name = $name,
            a.ageGroup = $age_group,
            a.gender = $gender,
            a.description = $description,
            a.embedding = $embedding

        WITH a
        MERGE (r:Role {name: $role})
        MERGE (a)-[:HAS_ROLE]->(r)
        """

        place_query = """
        MATCH (a:Character {id: $agent_id})
        MATCH (p:Place {id: $place_id})
        MERGE (a)-[:LIVES_IN]->(p)
        """

        trait_query = """
        MERGE (a:Character {id: $agent_id})
        MERGE (tr:Trait {name: $trait})
        MERGE (a)-[r:HAS_TRAIT]->(tr)
        SET r.strength = $strength
        """

        def build_agent_embedding_text(agent: Agent, folktale: Folktale) -> str:
            personality_dict = agent.personality.model_dump()

            personality_text = "\n".join(
                f"- {key.title()}: {value}"
                for key, value in personality_dict.items()
            )

            return f"""Folktale: {folktale.title}

Character: {agent.name}
Role in story: {agent.role}
Race: {agent.race}
Gender: {agent.gender}
Age group: {agent.age_group}

Personality traits:
{personality_text}

Description:
{agent.description}
"""

        descriptions = [
            build_agent_embedding_text(agent, folktale)
            for agent in agents
        ]
        
        embeddings = self.model.embed_documents(descriptions)

        for idx, agent in enumerate(agents):
            agent_id = agent.id

            params = {
                "agent_id": agent_id,
                "role": agent.role,
                "race": agent.race,
                "name": agent.name,
                "age_group": agent.age_group,
                "description": agent.description,
                "gender": agent.gender,
                "embedding": embeddings[idx]
            }

            self.neo4j.execute_query(agent_query, params)

            if agent.lives_in:
                self.neo4j.execute_query(place_query, {
                    "agent_id": agent_id,
                    "place_id": agent.lives_in
                })

            personality = agent.personality.model_dump()
            for trait, strength in personality.items():
                self.neo4j.execute_query(trait_query, {
                    "agent_id": agent_id,
                    "trait": trait,
                    "strength": float(strength)
                })
                
    def insert_relationships(self, relationships: list[Relationship]):
        query = """
        MATCH (a:Character {id: $source_id})
        MATCH (b:Agent {id: $target_id})
        MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b)
        SET r.description = $description,
            r.strength = $strength
        """

        for rel in relationships:
            self.neo4j.execute_query(query, {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.type,
                "description": rel.description,
                "strength": rel.strength
            })
    
    def insert_events(self, events: list[Event], folktale: Folktale):
        event_query = """
        MERGE (e:Event {id: $event_id})
        SET e.description = $description,
            e.name = $name,
            e.order = $order,
            e.thoughts = $thoughts,
            e.embedding = $embedding

        WITH e
        MATCH (p:Place {id: $place_id})
        MERGE (e)-[:TAKES_PLACE_IN]->(p)

        WITH e
        MERGE (fu:Function {name: $function})
        MERGE (e)-[:HAS_FUNCTION]->(fu)

        WITH e
        MATCH (f:Folktale {url: $url})
        MERGE (f)-[:HAS_EVENT]->(e)

        WITH f, e
        WHERE $is_first = true
        MERGE (f)-[:FIRST_EVENT]->(e)
        """

        agent_event_query = """
        MATCH (e:Event {id: $event_id})
        MATCH (a:Character {id: $agent_id})
        MERGE (e)-[r:HAS_CHARACTER]->(a)
        SET r.importance = $importance,
            r.actions = $actions
        """

        object_event_query = """
        MATCH (e:Event {id: $event_id})
        MATCH (o:Object {id: $object_id})
        MERGE (e)-[:HAS_OBJECT]->(o)
        """

        event_order_query = """
        MATCH (e1:Event {id: $prev_id})
        MATCH (e2:Event {id: $current_id})
        MERGE (e1)-[:POST_EVENT]->(e2)
        MERGE (e2)-[:PRE_EVENT]->(e1)
        """

        def build_event_embedding_text(event: Event, folktale: Folktale, order: int):
            return f"""Folktale: {folktale.title}

Event: {event.name}
Order in story: {order}
Narrative function: {event.type}

Place: {event.place}

Description:
{event.description}
"""

        descriptions = [
            build_event_embedding_text(event, folktale, idx)
            for idx, event in enumerate(events)
        ]

        embeddings = self.model.embed_documents(descriptions)

        for idx, event in enumerate(events):
            self.neo4j.execute_query(event_query, {
                "event_id": event.id,
                "description": event.description,
                "embedding": embeddings[idx],
                "name": event.name,
                "order": idx,
                "thoughts": event.thoughts,
                "place_id": event.place,
                "function": event.type,
                "url": folktale.url,
                "is_first": idx == 0
            })

            for agent in event.agents:
                self.neo4j.execute_query(agent_event_query, {
                    "event_id": event.id,
                    "agent_id": agent.id,
                    "importance": agent.importance,
                    "actions": agent.actions
                })

            for obj_id in event.objects:
                self.neo4j.execute_query(object_event_query, {
                    "event_id": event.id,
                    "object_id": obj_id
                })

        for i in range(1, len(events)):
            self.neo4j.execute_query(event_order_query, {
                "prev_id": events[i - 1].id,
                "current_id": events[i].id
            })

    def find_loose_entities(self):
        query = """
        CALL {
            MATCH (o:Object)
            WHERE NOT EXISTS {
                MATCH (o)<-[:HAS_OBJECT]-(:Event)
            }
            RETURN 'Object' AS type, o.id AS id, o.name AS name,
                'No event link' AS reason, o AS node

            UNION ALL

            MATCH (a:Character)
            WHERE NOT EXISTS {
                MATCH (a)<-[:HAS_CHARACTER]-(:Event)
            }
            RETURN 'Agent' AS type, a.id AS id, a.name AS name,
                'No event participation' AS reason, a AS node

            UNION ALL

            MATCH (p:Place)
            WHERE NOT EXISTS {
                MATCH (p)<-[:TAKES_PLACE_IN]-(:Event)
            }
            AND NOT EXISTS {
                MATCH (p)<-[:LIVES_IN]-(:Agent)
            }
            RETURN 'Place' AS type, p.id AS id, p.name AS name,
                'No event usage and unlinked' AS reason, p AS node
        }
        RETURN type, id, name, reason
        """

        return self.neo4j.execute_query(query)
    
    def delete_loose_entities(self):
        query = """
        CALL {
            MATCH (o:Object)
            WHERE NOT EXISTS { MATCH (o)<-[:HAS_OBJECT]-(:Event) }
            WITH collect(DISTINCT o) AS nodes
            WITH nodes, [n IN nodes | n.name] AS names
            UNWIND nodes AS o
            DETACH DELETE o
            RETURN 'Object' AS type, names

            UNION ALL

            MATCH (a:Character)
            WHERE NOT EXISTS { MATCH (a)<-[:HAS_CHARACTER]-(:Event) }
            WITH collect(DISTINCT a) AS nodes
            WITH nodes, [n IN nodes | n.name] AS names
            UNWIND nodes AS a
            DETACH DELETE a
            RETURN 'Agent' AS type, names

            UNION ALL

            MATCH (p:Place)
            WHERE NOT EXISTS { MATCH (p)<-[:TAKES_PLACE_IN]-(:Event) }
            AND NOT EXISTS { MATCH (p)<-[:LIVES_IN]-(:Agent) }
            WITH collect(DISTINCT p) AS nodes
            WITH nodes, [n IN nodes | n.name] AS names
            UNWIND nodes AS p
            DETACH DELETE p
            RETURN 'Place' AS type, names
        }
        RETURN type, names
        """
        return self.neo4j.execute_query(query)
    
    def setup(self):
        self.create_constraints()
        self.load_collections()
        self.load_structures()
    
    def add_folktale(self, folktale: Folktale):
        self.insert_folktale(folktale)
        self.insert_entities("Object", folktale.objects, folktale)
        self.insert_entities("Place", folktale.places, folktale)
        self.insert_agents(folktale.agents, folktale)
        self.insert_relationships(folktale.relationships)
        self.insert_events(folktale.events, folktale)
