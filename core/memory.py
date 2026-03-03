"""
Memory module for managing short-term context and persistent history via ChromaDB.
"""
import chromadb
import uuid
from typing import List, Dict, Callable, Optional

class Memory:
    """Class to manage short-term context and long-term persistent history."""

    def __init__(self, max_history: int = 20, 
                 embedding_fn: Optional[Callable[[str], List[float]]] = None, 
                 db_path: Optional[str] = "./chroma_db",
                 collection_name: str = "brain_memory"):
        """
        Initializes memory systems.
        
        Args:
            max_history: Maximum number of interactions for short-term memory.
            embedding_fn: Function to generate embeddings for text.
            db_path: Path to store local ChromaDB. If None, uses an ephemeral in-memory client.
            collection_name: The name of the collection in ChromaDB.
        """
        # Short-term memory
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.embedding_fn = embedding_fn
        
        # Long-term memory (ChromaDB)
        if db_path:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()
            
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def add_interaction(self, role: str, content: str) -> None:
        """Adds a new interaction to the short-term memory history."""
        self.history.append({"role": role, "content": content})
        
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def store_memory(self, text: str, metadata: Dict[str, str] = None) -> None:
        """
        Vectorizes and saves an interaction into long-term memory.
        
        Args:
            text: The text string to remember.
            metadata: Additional data (like role, emotion, etc).
        """
        doc_id = str(uuid.uuid4())
        
        if self.embedding_fn:
            embedding = self.embedding_fn(text)
            if not embedding:
                return
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
        else:
            # Let ChromaDB use its default SentenceTransformers embedding function
            # This is particularly useful for local testing without an API key
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )

    def retrieve_relevant_memories(self, query: str = None, top_k: int = 3, where: dict = None) -> List[Dict]:
        """
        Retrieves raw memory dictionaries (including text and metadata) from ChromaDB.
        
        Args:
            query: Semantic search query. If None, queries by metadata.
            top_k: Number of documents to fetch.
            where: Metadata dictionary to filter by.
        """
        if self.collection.count() == 0:
            return []
            
        actual_k = min(top_k, self.collection.count())
        query_kwargs = {"n_results": actual_k}
        
        if where:
            query_kwargs["where"] = where
            
        if query:
            if self.embedding_fn:
                query_embedding = self.embedding_fn(query)
                if not query_embedding:
                    return []
                query_kwargs["query_embeddings"] = [query_embedding]
            else:
                query_kwargs["query_texts"] = [query]
                
            results = self.collection.query(**query_kwargs)
            
            if not results['documents'] or not results['documents'][0]:
                return []
                
            ret = []
            for i in range(len(results['documents'][0])):
                ret.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })
            return ret
        else:
            # If there's no semantic query but we search by metadata filtering
            results = self.collection.get(where=where, limit=actual_k)
            ret = []
            for i in range(len(results['ids'])):
                ret.append({
                    "id": results['ids'][i],
                    "text": results['documents'][i],
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                })
            return ret

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Searches for similar past experiences and returns a formatted string.
        """
        memories = self.retrieve_relevant_memories(query=query, top_k=top_k)
        if not memories:
            return ""
        return "\n".join([f"- {m['text']}" for m in memories])

    def retrieve_by_emotion(self, emotion: str, top_k: int = 10) -> List[Dict]:
        """
        Filters memories based exclusively on a specific emotion.
        """
        return self.retrieve_relevant_memories(where={"emotion": emotion}, top_k=top_k)

    def get_context(self) -> str:
        """Retrieves the short-term conversation history."""
        if not self.history:
            return "No previous context."
        
        context_lines = []
        for entry in self.history:
            context_lines.append(f"{entry['role'].capitalize()}: {entry['content']}")
        
        return "\n".join(context_lines)

    def build_prompt(self, new_input: str, long_term_context: str = "", 
                     current_emotion: str = "Neutral", sensory_snapshot: Dict[str, str] = None) -> str:
        """
        Merges short-term context, long-term memory, emotional state, sensory data, and new input into a final prompt.
        """
        short_term_context = self.get_context()
        
        prompt = (
            f"Current Internal State:\n"
            f"- Dominant Emotion: {current_emotion}\n"
        )
        
        if sensory_snapshot:
            active_senses = [f"{k.capitalize()}: {v}" for k, v in sensory_snapshot.items() if v.lower() != 'none']
            if active_senses:
                prompt += f"- Active Senses: {', '.join(active_senses)}\n"
            else:
                prompt += f"- Active Senses: None detected.\n"
                
        prompt += f"\nContext (Recent):\n{short_term_context}\n\n"
        
        if long_term_context:
            prompt += f"Long-term Memories:\n{long_term_context}\n\n"
            
        prompt += (
            f"New Input:\nUser: {new_input}\n\n"
            f"Response:"
        )
        return prompt

    def clear(self) -> None:
        """Clears the short-term memory."""
        self.history = []
