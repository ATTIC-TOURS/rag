from llama_index.core.schema import TextNode
from .context_embedder import ContextEmbedderLLM


class ContextAugmentNodeProcessor:
    def __init__(self, whole_text: str, model_name: str):
        self.whole_text = whole_text
        self.context_embedder = ContextEmbedderLLM(model_name=model_name)

    def process_node(self, node: TextNode) -> TextNode:
        # Store the local chunk in metadata
        node.metadata["chunk_text"] = node.text
        # Augment the node text with the whole PDF context
        context_augment_text = self.context_embedder.embed_context(self.whole_text, node.text)
        node.text = context_augment_text
        return node