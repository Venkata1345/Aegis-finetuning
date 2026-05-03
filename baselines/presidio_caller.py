"""Microsoft Presidio baseline — rule-based PII detection.

Presidio outputs structured spans directly (no JSON to parse), so we map
Presidio entity types to Aegis categories and serialize as a JSON array.
Schema validity is True by construction.

Note on category mapping limitations (documented in README):
- Presidio's `DATE_TIME` doesn't disambiguate DOB from any date. We map
  it to DOB anyway as the closest fit, knowing this will produce DOB
  false positives that should appear in the eval-error analysis.
- `LOCATION` collapses cities/countries/streets into one category, which
  matches our ADDRESS umbrella.
- Several Presidio categories (URL, NRP, MEDICAL_LICENSE) don't cleanly
  map; we drop the ones outside our 9 categories.
"""

from __future__ import annotations

import json

from baselines.base import Predictor, Pricing
from data.schema import PIIType

# Presidio entity → Aegis category
_PRESIDIO_TO_AEGIS: dict[str, PIIType] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "ADDRESS",
    "DATE_TIME": "DOB",  # over-broad; documented
    "US_SSN": "GOV_ID",
    "US_PASSPORT": "GOV_ID",
    "US_DRIVER_LICENSE": "GOV_ID",
    "US_ITIN": "GOV_ID",
    "UK_NHS": "MEDICAL_ID",
    "MEDICAL_LICENSE": "MEDICAL_ID",
    "CREDIT_CARD": "FINANCIAL",
    "IBAN_CODE": "FINANCIAL",
    "CRYPTO": "FINANCIAL",
    "IP_ADDRESS": "IP_ADDRESS",
}


class PresidioPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__(name="presidio", pricing=Pricing())  # self-hosted, no cost
        from presidio_analyzer import AnalyzerEngine

        self.engine = AnalyzerEngine()

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        results = self.engine.analyze(text=text, language="en")
        spans = []
        for r in results:
            type_ = _PRESIDIO_TO_AEGIS.get(r.entity_type)
            if type_ is None:
                continue
            spans.append({
                "type": type_,
                "start": r.start,
                "end": r.end,
                "text": text[r.start : r.end],
            })
        return json.dumps(spans, ensure_ascii=False), None, None
