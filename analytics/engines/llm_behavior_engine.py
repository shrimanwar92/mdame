import json

import sys
import os
from .prompt import PROMPT, TOOL_SCHEMA
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import (
    MODEL_ID, get_bedrock_client, UNIVERSAL_PERSONAS_PATH
)


class LLMBehaviorEngine:
    def __init__(self):
        self.client = get_bedrock_client()

    def build_prompt(self, payload):
        return (
            PROMPT + "\n" +
            f"""
                UNIVERSAL MACRO PERSONA:{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
            """
        )

    def generate(self, macro_persona):
        prompt = self.build_prompt(macro_persona)

        response = self.client.converse(
            modelId=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [{
                    "text": prompt
                }]
            }],
            toolConfig={
                "tools": [{
                    "toolSpec": {
                        "name": "submit_product_behaviors",
                        "description": "Return product specific customer behaviors.",
                        "inputSchema": {
                            "json": TOOL_SCHEMA
                        }
                    }
                }],
                "toolChoice": {
                    "tool": {
                        "name": "submit_product_behaviors"
                    }
                }
            },
            inferenceConfig={
                "temperature": 0.0,
                "maxTokens": 4096
            }
        )

        for block in response["output"]["message"]["content"]:
            if "toolUse" not in block:
                continue
            tool_input = block["toolUse"]["input"]
            return tool_input["customer_behavior"]
