"""HELIOS visual outcome verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image
from transformers import pipeline


Verdict = Literal["SUCCESS", "FAILURE", "UNCERTAIN"]


@dataclass
class VerificationResult:
    verdict: Verdict
    raw_response: str


class VisualOutcomeVerifier:
    """Vision-language verifier operating only on rendered RGB observations."""

    MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def __init__(self, device: int = 0) -> None:
        print(f"Loading visual verifier: {self.MODEL_ID}")

        self.pipe = pipeline(
            "image-text-to-text",
            model=self.MODEL_ID,
            device=device,
            dtype=torch.float16,
        )

    @staticmethod
    def _extract_generated_text(output: object) -> str:
        if isinstance(output, str):
            return output

        if isinstance(output, list):
            if not output:
                return ""

            item = output[-1]

            if isinstance(item, str):
                return item

            if isinstance(item, dict):
                for key in ("generated_text", "text", "content"):
                    if key in item:
                        return VisualOutcomeVerifier._extract_generated_text(
                            item[key]
                        )

        if isinstance(output, dict):
            for key in ("generated_text", "text", "content"):
                if key in output:
                    return VisualOutcomeVerifier._extract_generated_text(
                        output[key]
                    )

        return str(output)

    def verify(
        self,
        instruction: str,
        before_image: Image.Image,
        middle_image: Image.Image,
        after_image: Image.Image,
    ) -> VerificationResult:
        prompt = f"""
You are evaluating the outcome of a robotic manipulation attempt.

The images are chronological:

IMAGE 1 = scene before execution.
IMAGE 2 = scene during execution.
IMAGE 3 = final scene after execution.

Robot instruction:
"{instruction}"

Your task is to determine whether the FINAL scene actually satisfies
the instruction.

Important:
- Identify the target object.
- Identify the required destination or spatial relationship.
- Do not call the attempt successful merely because the robot moved.
- If the target object remains outside the required destination, FAILURE.
- If the required final relationship is visibly satisfied, SUCCESS.
- If the final state cannot be determined reliably, UNCERTAIN.
- Be skeptical: success requires positive visual evidence.

Respond in exactly this format:

TARGET: <brief description>
FINAL_STATE: <brief description>
VERDICT: SUCCESS

The final verdict must be exactly one of:
SUCCESS
FAILURE
UNCERTAIN
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": before_image.convert("RGB"),
                    },
                    {
                        "type": "image",
                        "image": middle_image.convert("RGB"),
                    },
                    {
                        "type": "image",
                        "image": after_image.convert("RGB"),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        output = self.pipe(
            text=messages,
            max_new_tokens=80,
            return_full_text=False,
        )

        raw = self._extract_generated_text(output).strip()

        match = re.search(
            r"VERDICT\s*:\s*(SUCCESS|FAILURE|UNCERTAIN)",
            raw,
            flags=re.IGNORECASE,
        )

        if match is None:
            verdict: Verdict = "UNCERTAIN"
        else:
            verdict = match.group(1).upper()  # type: ignore[assignment]

        return VerificationResult(
            verdict=verdict,
            raw_response=raw,
        )
