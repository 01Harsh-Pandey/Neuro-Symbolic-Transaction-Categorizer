import yaml
import re
import os
from typing import Dict, Optional, List

class RuleEngine:
    def __init__(self, taxonomy_path="data/taxonomy.yaml"):
        self.taxonomy_path = taxonomy_path
        self.rules = self._load_rules()
        print(f"⚡ RULES: Loaded {len(self.rules)} regex patterns.")

    def _load_rules(self):
        if not os.path.exists(self.taxonomy_path):
            return []
        
        with open(self.taxonomy_path, "r") as f:
            data = yaml.safe_load(f)

        compiled = []
        for cat in data['taxonomy']:
            for sub in cat['subcategories']:
                for kw in sub['keywords']:
                    # Boundary check + Case insensitive
                    pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                    compiled.append({
                        "pattern": pattern, 
                        "category": cat['name'], 
                        "subcategory": sub['name'], 
                        "keyword": kw
                    })
        return compiled

    def apply(self, text: str) -> Optional[Dict]:
        """Apply regex rules to text, return match if found"""
        for r in self.rules:
            if r["pattern"].search(text):
                return {
                    "category": r["category"],
                    "subcategory": r["subcategory"],
                    "confidence": 1.0,
                    "source": "RULE",
                    "reason": f"Matched keyword: '{r['keyword']}'"
                }
        return None

    def get_rule_stats(self) -> Dict:
        """Return statistics about loaded rules"""
        categories = {}
        for rule in self.rules:
            cat = rule['category']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            "total_rules": len(self.rules),
            "rules_by_category": categories
        }