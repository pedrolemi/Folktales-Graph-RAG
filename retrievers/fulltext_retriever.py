from typing import Any, Optional
from neo4j_manager import Neo4jManager
from .base_retriever import BaseRetriever

class FullTextRetriever(BaseRetriever):
    def __init__(self, neo4j_manager: Neo4jManager, index_name: str, node_label: str, text_property: str, return_fields: Optional[dict[str, str]]):
        super().__init__(neo4j_manager, index_name, return_fields)
        self.node_label = node_label
        self.text_property = text_property
        
        self._create_fulltext_index()

    def _create_fulltext_index(self):
        """Crea un índice de texto completo."""
        query = f"""
        CREATE FULLTEXT INDEX {self.index_name} IF NOT EXISTS
        FOR (n:{self.node_label})
        ON EACH [n.{self.text_property}]
        """
        try:
            self.neo4j.execute_query(query)
        except Exception as e:
            print(f"Fulltext index already exists or error: {e}")

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """
        Recupera chunks usando búsqueda vectorial pura de texto completo.
        """
        # top_k = top_k or self.settings.top_k_results

        return_clause = self._build_return_clause("node")

        cypher_query = f"""
        // Fulltext search
        CALL db.index.fulltext.queryNodes($index_name, $query, {{limit: $top_k}})
        YIELD node, score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN {return_clause}
        """

        params = {
            "index_name": self.index_name,
            "query": query,
            "top_k": top_k
        }

        return self.neo4j.execute_query(cypher_query, params)
    