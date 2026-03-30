
from llama_index.vector_stores.postgres import PGVectorStore
from raglab.db_config import DB_CONFIG

class PGVectorStoreFactory:
    @staticmethod
    def create(table_name: str = "handbook_chunks") -> PGVectorStore:
        return PGVectorStore.from_params(
            database=DB_CONFIG["database"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            table_name=table_name,
            embed_dim=DB_CONFIG["embed_dim"],
        )
