from typing import Optional, Any
from neo4j_manager import Neo4jManager

class BaseRetriever:
    def __init__(self, neo4j_manager: Neo4jManager, index_name: str, return_projection: Optional[dict[str, str]] = None, extra_match: str = "", include_score: bool = True):
        self.neo4j = neo4j_manager
        self.index_name = index_name
        self.include_score = include_score
        self.extra_match = extra_match

        self.return_projection = return_projection or {
            "description": "node.description",
            "chunk_id": "node.id"
        }

    def _build_return_clause(self):
        fields = []

        for alias, expr in self.return_projection.items():
            fields.append(f"{expr} AS {alias}")

        if self.include_score:
            fields.append("score")

        return ", ".join(fields)

    def retrieve(self, query: str, top_k: Optional[int] = 2) -> list[dict[str, Any]]:
        raise NotImplementedError
