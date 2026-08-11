import nodes
from .minimaxh3 import MiniMaxH3HybridLoader

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    'MiniMaxH3HybridLoader': MiniMaxH3HybridLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'MiniMaxH3HybridLoader': 'MiniMax H3 Hybrid Loader',
}

print("\033[34mMinimax Hybrid Loader: \033[92mloaded\033[0m!\n")
