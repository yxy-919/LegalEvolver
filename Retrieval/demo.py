"""Exercise all four tools against the bundled database subsets."""
import os
from pathlib import Path
import json


def main():
    os.environ['RUBRIC_REACT_CONFIG'] = str(Path(__file__).with_name('demo_config.yaml'))
    from Retrieval.tool_manager import ToolManager
    tools = ToolManager()
    for name, query in [
        ('statute_search', '盗窃罪 犯罪构成'),
        ('case_search', '故意伤害 轻伤'),
        ('judicial_interpretation_search', '盗窃 数额认定'),
        ('guideline_search', '盗窃罪 量刑 从轻处罚'),
    ]:
        result = tools.call_tool(name, {'query': query, 'topk': 2})
        cards = result.get('results', result.get('national_guideline_evidence', []))
        if result.get('error') or not cards:
            raise RuntimeError(f'{name}: {result.get("error") or "no results"}')
        print(json.dumps({'tool': name, 'hits': len(cards), 'first_title': cards[0]['title']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
