from utils.regex_utils import snake_case_to_pascal_case
from utils.loader import load_json_folder
from neo4j_manager import Neo4jManager
from config import get_settings
from typing import Optional
from schemas.folktale import Folktale
from schemas.agent import Agent, Personality
from schemas.relationship import Relationship
from schemas.event import Event
from utils.models import get_embeddings
import os


class FolktaleGraphBuilder:
    def __init__(self, metadata_dir: str = "./metadata"):
        self.metadata_dir = metadata_dir
        self.structure_dir = os.path.join(metadata_dir, "structure")
        self.collections_dir = os.path.join(metadata_dir, "collections")

        self.settings = get_settings()

        self.manager = Neo4jManager()
        self.model = get_embeddings()

    def clear_database(self):
        self.manager.execute_query("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        constraints = [
            # Cuento popular
            "CREATE CONSTRAINT folktale_url IF NOT EXISTS FOR (f:Folktale) REQUIRE f.url IS UNIQUE",

            # IDs
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
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
        self.manager.create_constraints(constraints)

    def create_vector_database(self):
        self.manager.create_vector_index(
            index_name="event_embeddings",
            label="Event",
            property_name="embedding",
            dimensions=768,
            similarity="cosine",
            overwrite=True
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
                self.manager.execute_query(query, {
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

        self.manager.execute_query(query, params)

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
        SET f.title = $title,

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

        self.manager.execute_query(query, params)
    
    def insert_entities(self, label: str, items: list):
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n.type = $type,
            n.name = $name,
            n.description = $description
        """

        for item in items:
            params = {
                "id": item["id"],
                "type": item["type"],
                "name": item["name"],
                "description": item["description"]
            }
            self.manager.execute_query(query, params)

    def insert_agents(self, agents: list[Agent]):
        agent_query = """
        MERGE (a:Agent {id: $agent_id})
        SET a.race = $race,
            a.name = $name,
            a.ageGroup = $age_group,
            a.gender = $gender,
            a.description = $description

        WITH a
        MERGE (r:Role {name: $role})
        MERGE (a)-[:HAS_ROLE]->(r)
        """

        place_query = """
        MATCH (a:Agent {id: $agent_id})
        MATCH (p:Place {id: $place_id})
        MERGE (a)-[:LIVES_IN]->(p)
        """

        trait_query = """
        MERGE (a:Agent {id: $agent_id})
        MERGE (tr:Trait {name: $trait})
        MERGE (a)-[r:HAS_TRAIT]->(tr)
        SET r.strength = $strength
        """

        for agent in agents:
            agent_id = agent.id

            params = {
                "agent_id": agent_id,
                "role": agent.role,
                "race": agent.race,
                "name": agent.name,
                "age_group": agent.age_group,
                "description": agent.description,
                "gender": agent.gender
            }

            self.manager.execute_query(agent_query, params)

            if agent.lives_in:
                self.manager.execute_query(place_query, {
                    "agent_id": agent_id,
                    "place_id": agent.lives_in
                })

            for trait, strength in zip(Personality.model_fields, agent.personality):
                self.manager.execute_query(trait_query, {
                    "agent_id": agent_id,
                    "trait": trait,
                    "strength": strength
                })

    def insert_relationships(self, relationships: list[Relationship]):
        query = """
        MATCH (a:Agent {id: $source_id})
        MATCH (b:Agent {id: $target_id})
        MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b)
        SET r.description = $description,
            r.strength = $strength
        """

        for rel in relationships:
            self.manager.execute_query(query, {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.type,
                "description": rel.description,
                "strength": rel.strength
            })
    
    def insert_events(self, events: list[Event], folktale_url: str):
        event_query = """
        MERGE (e:Event {id: $event_id})
        SET e.description = $description,
            e.name = $name,
            e.order = $order
            e.thoughts = $thoughts
            e.embedding = $embedding,

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
        MATCH (a:Agent {id: $agent_id})
        MERGE (e)-[r:HAS_AGENT]->(a)
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

        descriptions = [event.description for event in events]
        embeddings = self.model.embed_documents(descriptions)

        for idx, event in enumerate(events):

            self.manager.execute_query(event_query, {
                "event_id": event.id,
                "description": event.description,
                "embedding": embeddings[idx],
                "name": event.name,
                "order": idx,
                "thoughts": event.thoughts,
                "place_id": event.place,
                "function": event.type,
                "url": folktale_url,
                "is_first": idx == 0
            })

            for agent in event.agents:
                self.manager.execute_query(agent_event_query, {
                    "event_id": event.id,
                    "agent_id": agent.id,
                    "importance": agent.importance,
                    "actions": agent.actions
                })

            for obj_id in event.objects:
                self.manager.execute_query(object_event_query, {
                    "event_id": event.id,
                    "object_id": obj_id
                })

        for i in range(1, len(events)):
            self.manager.execute_query(event_order_query, {
                "prev_id": events[i - 1].id,
                "current_id": events[i].id
            })

    def find_loose_entities(self):
        query = """
        MATCH (o:Object)
        WHERE NOT (o)<-[:HAS_OBJECT]-(:Event)
        RETURN 'Object' AS type, o.id AS id, o.name AS name, 'No event link' AS reason

        UNION ALL

        MATCH (a:Agent)
        WHERE NOT (a)<-[:HAS_AGENT]-(:Event)
        RETURN 'Agent' AS type, a.id AS id, a.name AS name, 'No event participation' AS reason

        UNION ALL

        MATCH (p:Place)
        WHERE NOT (p)<-[:TAKES_PLACE_IN]-(:Event)
        AND NOT (p)<-[:LIVES_IN]-(:Agent)
        RETURN 'Place' AS type, p.id AS id, p.name AS name, 'No event usage and uninhabited' AS reason
        """

        return self.manager.execute_query(query)
    
    def setup(self):
        self.clear_database()
        self.create_constraints()
        self.create_vector_database()
        self.load_collections()
        self.load_structures()

    def finalize(self):
        self.manager.close()
    
    def add_folktale(self, folktale: Folktale):
        self.insert_folktale(folktale)
        self.insert_entities("Object", folktale.objects)
        self.insert_entities("Place", folktale.places)
        self.insert_agents(folktale.agents)
        self.insert_relationships(folktale.relationships)
        self.insert_events(folktale.events, folktale.url)
