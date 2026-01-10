"""Text cleaning and normalization utilities."""
import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Handles:
    - Whitespace normalization
    - Unicode normalization
    - PDF hyphenation fixes (hyphens at line breaks)
    """
    if not text:
        return ""
    
    text = text.strip()
    
    text = re.sub(r'-\s*\n\s*', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    import unicodedata
    return unicodedata.normalize('NFKC', text)

