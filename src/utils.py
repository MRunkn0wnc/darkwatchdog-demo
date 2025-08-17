# src/utils.py
from typing import List, Dict

def format_post_text(post: Dict) -> str:
    return f"{post.get('title','')}\n{post.get('content','')}"
