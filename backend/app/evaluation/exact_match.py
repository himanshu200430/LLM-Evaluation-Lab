import re
def normalize(text):
    """Lowercase, replace punctuation with spaces, collapse extra spaces."""
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(text.split())

def exact_match(answer, expected):
    """Strictest check: same text, same case."""
    return answer.strip() == expected.strip()

def normalized_match(answer, expected):
    """Same text after ignoring case and punctuation."""
    return normalize(answer) == normalize(expected)

def contains_match(answer,expected):
    
    """Expected answer appears as whole words inside the answer.
    'The capital is Paris.' contains 'Paris', but '196' does not contain '96'."""
    return f" {normalize(expected)} " in f" {normalize(answer)} "