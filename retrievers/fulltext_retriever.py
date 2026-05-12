from typing import Any, Optional
from graph.neo4j_manager import Neo4jManager
from .base_retriever import BaseRetriever

class FullTextRetriever(BaseRetriever):
    def __init__(self, neo4j_manager: Neo4jManager, index_name: str, node_label: str, text_property: str, return_projection: Optional[dict[str, str]], extra_match: str = "", include_score: bool = True):
        super().__init__(neo4j_manager, index_name, return_projection, extra_match, include_score)
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
        
        return_clause = self._build_return_clause()

        cypher_query = f"""
        // Fulltext search
        CALL db.index.fulltext.queryNodes($index_name, $query, {{limit: $top_k}})
        YIELD node, score
        ORDER BY score DESC
        LIMIT $top_k

        {self.extra_match}

        RETURN {return_clause}
        """

        params = {
            "index_name": self.index_name,
            "query": query,
            "top_k": top_k
        }

        return self.neo4j.execute_query(cypher_query, params)
    