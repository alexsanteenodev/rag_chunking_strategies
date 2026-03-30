import os
import logging
from typing import List
from llama_index.core import Document

logger = logging.getLogger("LocalRAGLab")

class MarkdownLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load(self) -> List[Document]:
        docs = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".md"):
                    try:
                        path = os.path.join(root, file)
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        docs.append(Document(text=content, metadata={"file_path": path}))
                    except Exception as e:
                        logger.error(f"Failed to load {file}: {e}")
        logger.info(f"Loaded {len(docs)} markdown files from {self.directory}")
        return docs
