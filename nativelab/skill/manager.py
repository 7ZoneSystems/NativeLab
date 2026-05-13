from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


SKILLS_DIR = Path("./localllm/skill")
SKILLS_FILE = SKILLS_DIR / "skills.json"

DEFAULT_SKILL: Dict[str, Any] = {
    "name": "",
    "description": "",
    "instructions": "",
    "enabled": True,
}


def load_skills() -> List[Dict[str, Any]]:
    if not SKILLS_FILE.exists():
        return []
    try:
        raw = json.loads(SKILLS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = raw.get("skills", raw if isinstance(raw, list) else [])
    return [_merge_skill(row) for row in rows if isinstance(row, dict)]


def save_skills(skills: List[Dict[str, Any]]) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    clean = [_merge_skill(skill) for skill in skills]
    SKILLS_FILE.write_text(
        json.dumps({"skills": clean}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    clean = _merge_skill(skill)
    name = str(clean.get("name", "")).strip()
    if not name:
        raise ValueError("Skill name is required")
    clean["name"] = name
    skills = load_skills()
    for i, row in enumerate(skills):
        if row.get("name") == name:
            skills[i] = clean
            save_skills(skills)
            return clean
    skills.append(clean)
    save_skills(skills)
    return clean


def delete_skill(name: str) -> None:
    save_skills([skill for skill in load_skills() if skill.get("name") != name])


def active_skills() -> List[Dict[str, Any]]:
    return [skill for skill in load_skills() if skill.get("enabled", True)]


def active_skill_context(*, max_chars: int = 12000) -> str:
    skills = active_skills()
    if not skills:
        return ""
    lines = [
        "Active NativeLab skills are available for this response.",
        "Use them only when they are relevant to the user's request.",
        "",
    ]
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        description = str(skill.get("description", "")).strip()
        instructions = str(skill.get("instructions", "")).strip()
        if not name:
            continue
        lines.append(f"Skill: {name}")
        if description:
            lines.append(f"Description: {description}")
        if instructions:
            lines.append("Instructions:")
            lines.append(instructions)
        lines.append("")
    text = "\n".join(lines).strip()
    return text[:max_chars]


def _merge_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    clean = deepcopy(DEFAULT_SKILL)
    clean.update(skill or {})
    clean["name"] = str(clean.get("name", "")).strip()
    clean["description"] = str(clean.get("description", "")).strip()
    clean["instructions"] = str(clean.get("instructions", "")).strip()
    clean["enabled"] = bool(clean.get("enabled", True))
    return clean
