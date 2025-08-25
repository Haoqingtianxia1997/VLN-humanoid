import json
from pathlib import Path
import re
from typing import Any, Tuple, List, Dict
from src.mistral_ai.mistral import Mistralmodel
from src.mistral_ai.prompts.vision_prompt import system_prompt, example, assistant_prompt
from src.transcribe.tts import run_tts, play_text_to_speech
from src.utils import safe_extract_json_and_response_for_vlm



def run_mistral_vlm(user_prompt: str, image_path) -> Tuple[bool, str]:
    """Run Mistral Vision Language Model (VLM) for image analysis"""
    client = Mistralmodel()
    local_image =image_path #input("📂 input local image path: ").strip()
    vlm_script = "./src/mistral_ai/scripts/vlm_script.txt"
    vision_resp = client.chat_with_vision(user_prompt, local_image, system_prompt=system_prompt, example=example, assistant_prompt=assistant_prompt)

    # extract JSON objects from the response
    found, response, full_json = safe_extract_json_and_response_for_vlm(vision_resp)
    # 保存 response 文本
    print("🔍 found the target:", found)
    print("🗣️  response:", response)
    print("📦 complete JSON:\n", json.dumps(full_json, indent=2, ensure_ascii=False))

    # 保存 txt
    Path("./src/mistral_ai/scripts/vlm_script.txt").write_text(response, encoding="utf-8")

    # 保存完整 JSON
    with open("./src/mistral_ai/scripts/vlm_script.json", "w", encoding="utf-8") as jf:
        json.dump(full_json, jf, ensure_ascii=False, indent=2)

    if not response and not found:
        play_text_to_speech("Sorry, I can't find that. Please try again.", language='en')
        return False, None, None
    elif not found:
        run_tts(vlm_script)         
        return False, None, None

    # 从 object 列表里取第 1 个 label
    objects = full_json.get("object", [])
    first_name = objects[0].get("name", "") if objects else ""
    first_label = objects[0].get("label", "") if objects else ""
    return True, first_name, first_label