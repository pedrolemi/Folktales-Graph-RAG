from typing import Optional, Any
from neo4j_manager import Neo4jManager
from utils.models import get_embeddings

class BaseRetriever:
    def __init__(self, neo4j_manager: Neo4jManager, index_name: str, return_fields: Optional[dict[str, str]] = None):
        self.neo4j = neo4j_manager
        self.index_name = index_name
        self.return_fields = return_fields

        self.return_fields = return_fields or {
            "description": "description",
            "id": "chunk_id"
        }

    def _build_return_clause(self, node_alias: str, include_score : bool = True):
        fields = []

        for prop, alias in self.return_fields.items():
            fields.append(f"{node_alias}.{prop} AS {alias}")
        else:
            fields.append(node_alias)

        if include_score:
            fields.append("score")

        return ", ".join(fields)

    def retrieve(self, query: str, top_k: Optional[int] = 2) -> list[dict[str, Any]]:
        raise NotImplementedError
