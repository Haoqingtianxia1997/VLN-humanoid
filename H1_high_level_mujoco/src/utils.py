import json
import re
from typing import Any, Tuple, List, Dict

def get_last_text_line(filepath):
    """
    read the last non-empty line from a text file, ignoring comments and timestamps
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Reverse search for the last non-empty line that is not a comment
    for line in reversed(lines):
        if ']' in line:
            # Remove timestamp part
            text = line.split(']', 1)[-1].strip()
            if text:
                return text
        elif line and not line.startswith('//'):
            return line
    return ""

def get_full_text(filepath):
    """
    Read all valid non-empty lines from a text file, ignoring comments and timestamps,
    and return the combined full text.
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
            if ']' in stripped:
                # Remove timestamp like [YYYY-MM-DD HH:MM:SS]
                text = stripped.split(']', 1)[-1].strip()
            else:
                text = stripped
            if text:
                lines.append(text)
    return ' '.join(lines)

def safe_extract_json_and_response_for_llm(text: str) -> tuple[str, list[dict]]:
    """
    提取完整 JSON（含 response 和 actions），返回 response 字符串和动作列表。
    尽量从模型输出中提取第一个合法 JSON 块。
    """
    try:
        # 清理 markdown 包裹
        text = re.sub(r"```json|```", "", text).strip()

        # 尝试整体解析
        parsed = json.loads(text)
        response = parsed.get("response", "")
        actions = parsed.get("actions", [])
        return response, [parsed]  # 保持 extract_json 接口兼容
    except Exception as e:
        print(f"[safe_extract_json_and_response] JSON 解析失败: {e}")
        return "", []
    

def safe_extract_json_and_response_for_vlm(data: Any) -> Tuple[bool, str, Dict]:
    """
    返回三个值:
      1. found     : bool
      2. response  : str
      3. full_json : dict  ←  完整 JSON
    """
    try:
        # ── 1. 转成 dict ─────────────────────────────
        if isinstance(data, dict):
            parsed = data
        else:
            text = re.sub(r"```json|```", "", str(data)).strip()
            text = re.sub(r"\bFalse\b", "false", text)
            text = re.sub(r"\bTrue\b",  "true",  text)
            parsed = json.loads(text)

        # ── 2. 提取字段 ───────────────────────────────
        found = bool(parsed.get("if_find", False))

        raw_resp = parsed.get("response", "")
        response = " ".join(raw_resp) if isinstance(raw_resp, list) else str(raw_resp)

        return found, response, parsed        # ← 直接返回完整 dict

    except Exception as e:
        print(f"[safe_extract_json_and_response] JSON 解析失败: {e}")
        return False, "", {}
