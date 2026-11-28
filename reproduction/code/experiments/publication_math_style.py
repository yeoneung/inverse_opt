"""Apply recorded typography changes to generated mathematical exposition.

Each exact math-span replacement has the same mathematical content. Numerical
calculations and stored data are unaffected. Already styled text is unchanged.
"""
import json
from pathlib import Path
import re

MATH = re.compile(r'(?<!\\)\$.*?(?<!\\)\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}', re.S)
RULES = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def apply(text, section):
    mapping = {row['before']:row['after'] for row in RULES.get(section,[])}
    return MATH.sub(lambda m:mapping.get(m[0],m[0]),text)
