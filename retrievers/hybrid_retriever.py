from neo4j_manager import Neo4jManager
from utils.models import get_embeddings
from .fulltext_retriever import FullTextRetriever
from typing import Optional, Any

class HybridRetriever(FullTextRetriever):
    def __init__(self, neo4j_manager: Neo4jManager, vector_index: str, fulltext_index: str, node_label: str, text_property: str, vector_weight: float = 0.5, return_fields: Optional[dict[str, str]] = None):
        super().__init__(neo4j_manager, fulltext_index, node_label, text_property, return_fields)
        self.vector_index = vector_index
        self.neo4j = neo4j_manager
        self.embedding_gen = get_embeddings()
        self.vector_weight = vector_weight
        # self.settings = get_settings()
        # self._create_fulltext_index()

    def retrieve(self, query: str, top_k: Optional[int] = 2) -> list[dict[str, Any]]:
        """
        Recupera chunks usando búsqueda híbrida (vectorial + texto completo).
        """
        # top_k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_gen.embed_query(query)

        return_clause = self._build_return_clause("node")

        cypher_query = f"""
        CALL () {{
            // Vector search
            CALL db.index.vector.queryNodes($vector_index, $top_k, $query_embedding)
            YIELD node, score
            WITH collect({{node: node, score: score}}) AS nodes, max(score) AS maxScore, min(score) AS minScore
            UNWIND nodes AS n
            RETURN n.node AS node,
                CASE
                    WHEN maxScore = minScore THEN 1.0
                    ELSE (n.score - minScore) / (maxScore - minScore)
                END * $vector_weight AS score

            UNION ALL

            // Fulltext search
            CALL db.index.fulltext.queryNodes($fulltext_index, $query, {{limit: $top_k}})
            YIELD node, score
            WITH collect({{node: node, score: score}}) AS nodes, max(score) AS maxScore, min(score) AS minScore
            UNWIND nodes AS n
            RETURN n.node AS node,
                CASE    
                    WHEN maxScore = minScore THEN 1.0
                    ELSE (n.score - minScore) / (maxScore - minScore)
                END * (1.0 - $vector_weight) AS score
        }}
        WITH node, sum(score) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN {return_clause}
        """

        params = {
            "vector_index": self.vector_index,
            "fulltext_index": self.index_name,
            "query_embedding": query_embedding,
            "query": query,
            "top_k": top_k,
            "vector_weight": self.vector_weight,
        }

        return self.neo4j.execute_query(cypher_query, params)
