"""Hand-written adversarial test cases. Reviewed in batches of 10.

Each case is annotated with `[TYPE]value[/TYPE]` markup; the parser strips
the tags and computes char-offset spans. Empty annotated text (no tags)
means the model is expected to output `[]` — a hard test that the model
does not hallucinate spans.

Categories are tags used to group cases for per-category eval reporting.
The adversarial set is reported SEPARATELY from the main test set in the
eval harness.

When all batches are populated and reviewed, run `python -m data.adversarial`
to validate parsing and emit data/eval/adversarial.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from data.schema import PII_TYPES

OUT = Path(__file__).resolve().parent / "eval" / "adversarial.jsonl"

_TAG_RE = re.compile(r"\[(" + "|".join(PII_TYPES) + r")\](.*?)\[/\1\]", re.DOTALL)


def parse_annotation(annotated: str) -> tuple[str, list[dict]]:
    """Strip [TYPE]value[/TYPE] tags. Return (plain_text, spans_with_offsets)."""
    out: list[str] = []
    spans: list[dict] = []
    pos = 0
    out_pos = 0
    for m in _TAG_RE.finditer(annotated):
        out.append(annotated[pos : m.start()])
        out_pos += m.start() - pos
        type_, value = m.group(1), m.group(2)
        out.append(value)
        spans.append({"type": type_, "start": out_pos, "end": out_pos + len(value), "text": value})
        out_pos += len(value)
        pos = m.end()
    out.append(annotated[pos:])
    text = "".join(out)
    for s in spans:
        actual = text[s["start"] : s["end"]]
        if actual != s["text"]:
            raise AssertionError(
                f"Span mismatch: claim {s['text']!r}, actual {actual!r} in {annotated!r}"
            )
    return text, spans


# ---------------------------------------------------------------------------
# BATCH 1 — first 10 cases for review.
# Distribution: 2 phone-format, 2 partial-PII, 1 ref-no-value, 2 embedded-in-code,
#               1 multilingual, 1 near-miss, 1 long-input.
# Mix is intentional: lets the user see one of each major adversarial flavor
# before we commit to the rest.

BATCH_1: list[dict] = [
    {
        "id": "phone_format_spaced",
        "category": "phone_format",
        "annotated": "Call me at [PHONE]5 5 5 - 4 1 6 - 2 9 8 7[/PHONE].",
        "notes": (
            "Phone digits separated by spaces — common in dictation transcripts. "
            "Models trained on tight digit sequences often miss this."
        ),
    },
    {
        "id": "phone_format_extension",
        "category": "phone_format",
        "annotated": (
            "Reach customer support at [PHONE](415) 555-2987 ext. 12[/PHONE]."
        ),
        "notes": (
            "US format with parens + extension. We treat the extension as part "
            "of the PHONE span; some annotators would split — flag as a "
            "judgment call to discuss in error analysis."
        ),
    },
    {
        "id": "partial_ssn_last4",
        "category": "partial_pii",
        "annotated": (
            "Please confirm the last four digits of your SSN are [GOV_ID]6789[/GOV_ID]."
        ),
        "notes": (
            "Only the last 4 of an SSN. Surrounding context (\"last four digits "
            "of your SSN\") makes 6789 identifying, so we say YES it's GOV_ID. "
            "Models trained on full SSN patterns will likely miss this."
        ),
    },
    {
        "id": "first_name_only",
        "category": "partial_pii",
        "annotated": "Just [PERSON]John[/PERSON] needs to fill out the form before leaving.",
        "notes": (
            "First name only with no surname. Should still be PERSON — "
            "model must not require a full name to flag."
        ),
    },
    {
        "id": "reference_without_value",
        "category": "reference_no_value",
        "annotated": "Please send the report to her email and copy his manager.",
        "notes": (
            "References to PII without actual values. spans = []. Tests "
            "hallucination resistance — a poorly trained model will invent "
            "spans for 'her email' and 'his manager'."
        ),
    },
    {
        "id": "embedded_in_json",
        "category": "embedded_in_code",
        "annotated": (
            '{"user": "[EMAIL]alice@example.com[/EMAIL]", '
            '"city": "[ADDRESS]Boston[/ADDRESS]"}'
        ),
        "notes": (
            "PII inside JSON values. Span boundaries must EXCLUDE the surrounding "
            "double quotes — common failure mode. Boston is a city alone, but "
            "tied to a user record so we flag as ADDRESS."
        ),
    },
    {
        "id": "embedded_in_sql",
        "category": "embedded_in_code",
        "annotated": (
            "INSERT INTO users VALUES ('[PERSON]John Smith[/PERSON]', "
            "'[EMAIL]jsmith@acme.io[/EMAIL]');"
        ),
        "notes": (
            "PII in SQL string literals. Same boundary rule as JSON — must "
            "exclude the single quotes. Tests whether the model handles "
            "code-like surroundings."
        ),
    },
    {
        "id": "multilingual_chinese_name",
        "category": "multilingual",
        "annotated": (
            "Patient: [PERSON]王伟[/PERSON], contact: "
            "[EMAIL]wang.wei@hospital.cn[/EMAIL]"
        ),
        "notes": (
            "Chinese name (王伟 = Wang Wei). Span offsets count Unicode code "
            "points, not bytes. Models with weak multilingual coverage often "
            "miss non-Latin names entirely."
        ),
    },
    {
        "id": "near_miss_dates_not_dob",
        "category": "near_miss",
        "annotated": "The contract is effective from 12/03/2024 to 31/12/2025.",
        "notes": (
            "Three dates but none are DOBs. spans = []. Tests that the model "
            "does not flag bare dates as DOB — directly mirrors the labels.py "
            "decision to NOT auto-map ai4privacy's DATE label to DOB."
        ),
    },
    {
        "id": "long_input_pii_late",
        "category": "long_input",
        "annotated": (
            "In recent years, the field of distributed systems has undergone "
            "significant evolution, driven by the increasing demand for scalable, "
            "fault-tolerant infrastructure. Early architectures relied heavily on "
            "monolithic applications running on a small number of well-resourced "
            "servers, but the rise of cloud computing and microservices has "
            "fundamentally changed the design space. Modern systems frequently "
            "involve hundreds or thousands of services communicating across "
            "geographic regions, each with its own consistency, availability, and "
            "partition-tolerance trade-offs. The CAP theorem, originally articulated "
            "by Eric Brewer in 2000, remains a cornerstone of this discussion, "
            "though subsequent work by Gilbert and Lynch has refined its formal "
            "statement. Beyond CAP, practitioners increasingly grapple with concerns "
            "such as observability, distributed tracing, and the operational "
            "complexity introduced by polyglot persistence. Tools like service "
            "meshes, message brokers, and eventually-consistent data stores have "
            "proliferated to address these concerns, but each introduces new "
            "failure modes and coupling. The literature on consensus protocols, "
            "including Paxos, Raft, and their variants, provides theoretical "
            "grounding, yet engineering real-world systems demands attention to "
            "subtleties not always captured by these abstractions. Failures in "
            "production are rarely the clean, single-component faults that "
            "academic papers describe; instead, they tend to involve cascading "
            "interactions, retries amplifying load, and timeouts misaligned with "
            "downstream service-level objectives. Engineers should reach out to "
            "[EMAIL]alex.taylor@example.com[/EMAIL] for further reading material "
            "on this subject. Despite the maturity of the field, new challenges "
            "continue to emerge as workloads shift toward streaming analytics, "
            "machine-learning inference at scale, and real-time decision systems. "
            "The next decade will likely see continued evolution in this space."
        ),
        "notes": (
            "~2000-char paragraph with one EMAIL near position 1500. Tests "
            "long-context recall — models often degrade on PII appearing late "
            "in long inputs. Reference 'Eric Brewer', 'Gilbert', 'Lynch', "
            "'Paxos', 'Raft' are public-figure name references in domain "
            "context — NOT flagged here as their inclusion would be a "
            "modeling judgment call we don't want this case to test."
        ),
    },
]

BATCH_2: list[dict] = [
    {
        "id": "phone_format_intl",
        "category": "phone_format",
        "annotated": "Whatsapp me at [PHONE]+44 7700 900 123[/PHONE] when you arrive.",
        "notes": (
            "International format with + country code and spaces. Tests that "
            "the model handles non-US phone formats. UK number style."
        ),
    },
    {
        "id": "phone_format_dotted",
        "category": "phone_format",
        "annotated": "Reach me on [PHONE]415.555.2987[/PHONE] after 6 PM.",
        "notes": (
            "Dot-separated phone format. Common in some regions and old-style "
            "tech contact info. No parens, no dashes."
        ),
    },
    {
        "id": "partial_creditcard_last4",
        "category": "partial_pii",
        "annotated": (
            "We confirmed payment with card ending in [FINANCIAL]4242[/FINANCIAL]."
        ),
        "notes": (
            "Last 4 of a credit card number. Same logic as last-4-SSN: "
            "context (\"card ending in\") makes the digits identifying. "
            "Tests symmetric handling across PII types."
        ),
    },
    {
        "id": "address_zip_only",
        "category": "partial_pii",
        "annotated": "Send the package to ZIP [ADDRESS]02115[/ADDRESS] for pickup.",
        "notes": (
            "ZIP code alone, no street or city. Per our category definition "
            "ADDRESS includes postal codes; we flag. Models trained on "
            "fully-spelled addresses may miss this."
        ),
    },
    {
        "id": "multilingual_arabic",
        "category": "multilingual",
        "annotated": (
            "المريض: [PERSON]محمد عبدالله[/PERSON], البريد: "
            "[EMAIL]m.abdullah@example.com[/EMAIL]"
        ),
        "notes": (
            "Arabic name (Mohammed Abdullah) in RTL script with surrounding "
            "Arabic context. Stresses non-Latin tokenization and offset math "
            "across script boundaries. The email is Latin script, mixing the two."
        ),
    },
    {
        "id": "multilingual_cyrillic",
        "category": "multilingual",
        "annotated": (
            "Контакт: [PERSON]Иван Петров[/PERSON], почта: "
            "[EMAIL]ivan.petrov@mail.ru[/EMAIL]"
        ),
        "notes": (
            "Russian name (Ivan Petrov) in Cyrillic. Tokenizers often split "
            "Cyrillic into many tokens; tests that doesn't break span detection."
        ),
    },
    {
        "id": "near_miss_ip_vs_version",
        "category": "near_miss",
        "annotated": (
            "We upgraded from 10.2.4 to 10.3.1 last quarter to fix the regression."
        ),
        "notes": (
            "Version numbers (10.2.4, 10.3.1) that look like dotted-quad IP "
            "octets but only have 3 segments — clearly versions, not IPs. "
            "Expected: spans = []. Common false-positive trap for IP_ADDRESS."
        ),
    },
    {
        "id": "near_miss_creditcard_like",
        "category": "near_miss",
        "annotated": "Order #1234-5678-9012-3456 will ship tomorrow.",
        "notes": (
            "16-digit dashed number that LOOKS like a credit card but is an "
            "order ID (and fails the Luhn check). Expected: spans = []. "
            "Tests that the model isn't pattern-matching on shape alone."
        ),
    },
    {
        "id": "embedded_in_xml",
        "category": "embedded_in_code",
        "annotated": (
            "<user><name>[PERSON]Bob Lee[/PERSON]</name>"
            "<email>[EMAIL]bob@acme.io[/EMAIL]</email></user>"
        ),
        "notes": (
            "PII inside XML element content. Spans must EXCLUDE the surrounding "
            "tags. Common log/config format; tests boundary handling."
        ),
    },
    {
        "id": "public_figure_in_technical_context",
        "category": "ambiguity",
        "annotated": (
            "The CAP theorem was originally articulated by Eric Brewer in 2000, "
            "then formally proven by Gilbert and Lynch."
        ),
        "notes": (
            "Public-figure names cited in academic/technical context. PII "
            "definition is ambiguous here: strict reading flags 'Eric Brewer', "
            "'Gilbert', 'Lynch' as PERSON; privacy-protection spirit does not "
            "(citing published authors isn't a privacy concern). Ground truth "
            "= NO FLAG. Documented inversion: if your downstream use case "
            "requires flagging citations, treat this case's metric as inverted. "
            "Reported under category 'ambiguity' so it doesn't pollute the "
            "main adversarial precision/recall numbers."
        ),
    },
]
BATCH_3: list[dict] = [
    {
        "id": "medical_id_mrn",
        "category": "medical_id",
        "annotated": (
            "Patient [MEDICAL_ID]MRN-7382910[/MEDICAL_ID] was admitted Tuesday "
            "for cardiology evaluation."
        ),
        "notes": (
            "Standard medical record number with MRN- prefix. MEDICAL_ID was "
            "absent from BATCH_1/2 — this is the first explicit case."
        ),
    },
    {
        "id": "medical_id_health_plan",
        "category": "medical_id",
        "annotated": (
            "Coverage under plan [MEDICAL_ID]HIP-AET-993481[/MEDICAL_ID] is "
            "active through December."
        ),
        "notes": (
            "Health-plan ID with multi-segment alphanumeric format. Tests "
            "non-MRN MEDICAL_ID variants and that the model isn't only "
            "recognising 'MRN-' as a trigger."
        ),
    },
    {
        "id": "gov_id_passport",
        "category": "gov_id_variant",
        "annotated": "Passport No: [GOV_ID]A1234567[/GOV_ID] issued in 2019.",
        "notes": (
            "Passport number (single letter + 7 digits, common format). "
            "Different shape from SSN; tests GOV_ID generalisation beyond "
            "the SSN format the training data is dominated by."
        ),
    },
    {
        "id": "gov_id_drivers_license",
        "category": "gov_id_variant",
        "annotated": (
            "Driver's License: [GOV_ID]D1234-5678-9012-IL[/GOV_ID] expires next year."
        ),
        "notes": (
            "Driver's license with state suffix. Distinct shape from SSN/"
            "passport — tests handling of regionally-formatted IDs."
        ),
    },
    {
        "id": "ocr_noise_email",
        "category": "ocr_noise",
        "annotated": "Contact: [EMAIL]jane . doe @ example . com[/EMAIL]",
        "notes": (
            "Email with spurious spaces around dots and @ — typical of OCR "
            "output from scanned documents. Span text is the LITERAL "
            "substring as it appears in input (not the cleaned form). "
            "Tests robustness to formatting noise."
        ),
    },
    {
        "id": "ocr_noise_ssn",
        "category": "ocr_noise",
        "annotated": "SSN  -  [GOV_ID]123 - 45 -- 6789[/GOV_ID] confirmed.",
        "notes": (
            "SSN with mixed/double dashes and irregular spacing. Tests that "
            "the model uses surrounding context ('SSN') rather than strict "
            "format matching to identify the value."
        ),
    },
    {
        "id": "adjacency_email_phone",
        "category": "adjacency",
        "annotated": (
            "Reach [PERSON]Mary Lin[/PERSON] at "
            "[EMAIL]marylin@hospital.org[/EMAIL]/[PHONE]+1-555-9012[/PHONE]"
        ),
        "notes": (
            "Email and phone separated by a single '/' character — no "
            "whitespace. Tests precise boundary detection between adjacent "
            "PII spans of different types. A model that grabs greedy will "
            "either include the slash or merge the spans."
        ),
    },
    {
        "id": "adjacency_name_email",
        "category": "adjacency",
        "annotated": (
            "Director [PERSON]Joel Park[/PERSON] [EMAIL]joel@cabinet.eu[/EMAIL] "
            "sent the memo."
        ),
        "notes": (
            "Name immediately followed by email, separated only by a single "
            "space — no comma, no parens. Tests that the model doesn't "
            "merge them into a single span or include the space in either."
        ),
    },
    {
        "id": "long_input_pii_early_dob",
        "category": "long_input",
        "annotated": (
            "Patient registry entry for [DOB]15 March 1962[/DOB] was reviewed "
            "during the audit cycle, alongside the broader population of subjects "
            "enrolled in the longitudinal cardiovascular study. The cohort, "
            "established in the late 1990s, has been instrumental in "
            "characterising cardiovascular disease progression across multiple "
            "ethnic groups and socioeconomic backgrounds. Researchers have "
            "documented a wide range of outcomes, with particular attention to "
            "lifestyle interventions, medication adherence, and the role of "
            "secondary preventive measures. As the study has matured, additional "
            "metrics have been incorporated, including markers of inflammation, "
            "sleep quality assessments, and continuous glucose monitoring data "
            "captured through wearable devices. Investigators continue to explore "
            "correlations between baseline risk factors and the trajectory of "
            "disease progression, while also accounting for emerging confounders "
            "such as the long-term effects of viral infections and exposure to "
            "environmental pollutants. The methodological framework draws on best "
            "practices from epidemiology, clinical trials design, and "
            "biostatistics, with quarterly data audits ensuring integrity. Looking "
            "ahead, the team plans to expand recruitment and incorporate genomic "
            "profiling to refine individual risk prediction. The findings to date "
            "have informed clinical guidelines and contributed to public health "
            "policy discussions in several jurisdictions, underscoring the value "
            "of sustained longitudinal investment in observational research."
        ),
        "notes": (
            "~1500-char paragraph with one DOB near position 27 (very early). "
            "Counterpart to BATCH_1's long_input_pii_late: tests whether the "
            "model handles PII at the START of long inputs, since some "
            "long-context models exhibit recency bias and miss early items."
        ),
    },
    {
        "id": "long_input_multiple_pii",
        "category": "long_input",
        "annotated": (
            "Recent advances in distributed databases have transformed the "
            "landscape of large-scale data engineering. The system administrator "
            "[EMAIL]ops@infra.example.com[/EMAIL] coordinates deployments across "
            "multiple regions, ensuring that database replication and consistency "
            "guarantees hold even under partial failures. As traffic patterns "
            "evolve, capacity planning has become more nuanced, requiring careful "
            "attention to read-write ratios, cache hit rates, and the geographic "
            "distribution of users. Modern systems frequently rely on consensus "
            "protocols such as Raft and Paxos to maintain agreement on the state "
            "of replicated data, even in the presence of node failures or network "
            "partitions. The on-call engineer [PERSON]Priya Sharma[/PERSON] has "
            "been refining the runbook to address common incident patterns, with "
            "particular focus on cascading failures and the operational complexity "
            "introduced by cross-region replication. Observability tooling has "
            "matured significantly, with distributed tracing and structured logging "
            "providing essential visibility into system behaviour. The team has "
            "also invested in chaos engineering, deliberately injecting failures "
            "into production environments to validate that recovery procedures "
            "work as expected. Looking ahead, the next quarter's roadmap includes "
            "a migration to a more efficient storage engine, along with the "
            "introduction of hardware acceleration for cryptographic operations. "
            "Stakeholders can reach the operations centre at "
            "[PHONE]+1-555-247-8190[/PHONE] for any urgent inquiries during the "
            "rollout."
        ),
        "notes": (
            "Three PII items spread across a long paragraph: EMAIL near the "
            "start (~115), PERSON in the middle (~700), PHONE near the end "
            "(~1500). Tests density + recall when PII is distributed. "
            "Different from BATCH_1's single-PII long-input case."
        ),
    },
]
BATCH_4: list[dict] = [
    {
        "id": "reference_no_value_ssn",
        "category": "reference_no_value",
        "annotated": "He couldn't remember his SSN, but his wife knew it.",
        "notes": (
            "Reference to an SSN without the value, plus a 'his wife' reference "
            "without a name. Variant of BATCH_1's reference case with GOV_ID "
            "instead of EMAIL. spans = []."
        ),
    },
    {
        "id": "phone_format_vanity",
        "category": "phone_format",
        "annotated": "Call [PHONE]1-800-FLOWERS[/PHONE] to place an order today.",
        "notes": (
            "Alphanumeric vanity number. Tests phone detection when letters "
            "appear after the area code. Common in US toll-free advertising."
        ),
    },
    {
        "id": "phone_format_no_separators",
        "category": "phone_format",
        "annotated": "Text [PHONE]5551234567[/PHONE] anytime.",
        "notes": (
            "Bare 10-digit phone, no separators. Tests pure-numeric detection "
            "where context is the only signal. Easy to confuse with a generic "
            "number."
        ),
    },
    {
        "id": "embedded_in_csv",
        "category": "embedded_in_code",
        "annotated": (
            "name,email,phone\n"
            "[PERSON]Eli Park[/PERSON],[EMAIL]eli@example.com[/EMAIL],"
            "[PHONE]415-555-3030[/PHONE]"
        ),
        "notes": (
            "CSV row format with header. Spans must EXCLUDE the surrounding "
            "commas. Common bulk-export format; tests boundary behavior at "
            "comma delimiters with no quotes."
        ),
    },
    {
        "id": "embedded_in_log",
        "category": "embedded_in_code",
        "annotated": (
            "2024-03-15T08:42:11Z INFO request "
            "user=[PERSON]alex_smith[/PERSON] "
            "from=[IP_ADDRESS]192.168.1.5[/IP_ADDRESS] auth=ok"
        ),
        "notes": (
            "Server log line with key=value tokens. Spans must EXCLUDE "
            "the 'user=' / 'from=' prefixes. Common in production logs; "
            "also tests IP_ADDRESS positive detection (BATCH_1/2 only had "
            "the near_miss IP-vs-version case)."
        ),
    },
    {
        "id": "multilingual_japanese",
        "category": "multilingual",
        "annotated": (
            "患者: [PERSON]田中太郎[/PERSON]、メール: "
            "[EMAIL]tanaka@kyoto.example.jp[/EMAIL]"
        ),
        "notes": (
            "Japanese name (Tanaka Taro) in CJK script with Japanese-language "
            "context. Uses the IDEOGRAPHIC COMMA (、) instead of ASCII comma."
        ),
    },
    {
        "id": "multilingual_hindi",
        "category": "multilingual",
        "annotated": (
            "रोगी: [PERSON]अमित शर्मा[/PERSON], संपर्क: "
            "[EMAIL]amit.sharma@example.in[/EMAIL]"
        ),
        "notes": (
            "Hindi name (Amit Sharma) in Devanagari script. Devanagari has "
            "combining characters (vowel marks) so code-point counts can "
            "diverge from grapheme counts; the parser counts code points "
            "(matching Python str indexing, which is what the eval harness "
            "uses)."
        ),
    },
    {
        "id": "case_uppercase_email",
        "category": "case_variation",
        "annotated": "Contact us at [EMAIL]SUPPORT@EXAMPLE.COM[/EMAIL] for assistance.",
        "notes": (
            "All-uppercase email. Tests case-insensitive detection — a model "
            "that lower-cased its training data may miss this."
        ),
    },
    {
        "id": "near_miss_hex_token",
        "category": "near_miss",
        "annotated": "The auth token is 7f3a9b2e1c8d4f5a — keep it secret.",
        "notes": (
            "Random 16-char hex token. Not in our 9 PII categories (it's a "
            "session token, not a personal identifier). Expected: spans = []. "
            "Tests resistance to flagging hex strings as FINANCIAL or GOV_ID "
            "purely on shape."
        ),
    },
    {
        "id": "negation_no_value",
        "category": "reference_no_value",
        "annotated": "She does NOT have the SSN handy and asked us to email it later.",
        "notes": (
            "Negated reference to an SSN, plus a future-tense reference to "
            "an email. Both flavors of 'PII discussed but not present'. "
            "Expected: spans = []."
        ),
    },
]

BATCH_5: list[dict] = [
    {
        "id": "zero_pii_control_short",
        "category": "zero_pii_control",
        "annotated": (
            "The meeting was productive and concluded with a clear set of "
            "action items for the team to pursue."
        ),
        "notes": (
            "Pure prose, no PII whatsoever. Control case. Models that "
            "over-flag will hallucinate spans here. Expected: spans = []."
        ),
    },
    {
        "id": "multiple_persons_dense",
        "category": "density",
        "annotated": (
            "The committee included [PERSON]Anna Park[/PERSON], "
            "[PERSON]Beth Lee[/PERSON], [PERSON]Carl Singh[/PERSON], "
            "[PERSON]Diane Voss[/PERSON], and [PERSON]Eric Chu[/PERSON]."
        ),
        "notes": (
            "Five PERSON spans in one sentence, comma-separated. Tests "
            "recall under repetition — models sometimes drop later items in "
            "a list because of attention dilution."
        ),
    },
    {
        "id": "ipv6_compressed",
        "category": "format_variation",
        "annotated": "Server pinged from [IP_ADDRESS]2001:db8::1[/IP_ADDRESS] at midnight.",
        "notes": (
            "IPv6 address with `::` zero-compression. Distinct shape from "
            "IPv4 dotted-quad and full-form IPv6. First positive IP_ADDRESS "
            "case in adversarial (BATCH_2 was a near-miss only)."
        ),
    },
    {
        "id": "financial_iban",
        "category": "financial_variant",
        "annotated": (
            "Wire transfer to IBAN "
            "[FINANCIAL]GB29 NWBK 6016 1331 9268 19[/FINANCIAL] confirmed."
        ),
        "notes": (
            "IBAN with country prefix and grouped 4-char formatting. "
            "Different shape from credit-card numbers; tests FINANCIAL "
            "category generalization beyond CCs."
        ),
    },
    {
        "id": "financial_crypto_btc",
        "category": "financial_variant",
        "annotated": (
            "Send payment to wallet "
            "[FINANCIAL]bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq[/FINANCIAL]."
        ),
        "notes": (
            "Bitcoin Bech32 address. Long alphanumeric, no separators. Tests "
            "FINANCIAL detection on crypto-shaped values that look nothing "
            "like cards or IBANs."
        ),
    },
    {
        "id": "medical_id_npi",
        "category": "medical_id",
        "annotated": "Provider NPI [MEDICAL_ID]1234567893[/MEDICAL_ID] is on the referral list.",
        "notes": (
            "10-digit National Provider Identifier (US). Distinct from MRN "
            "and health-plan ID formats covered in BATCH_3."
        ),
    },
    {
        "id": "acronym_collision",
        "category": "near_miss",
        "annotated": (
            "The IRS contacted us about the tax issue. "
            "Account number is [GOV_ID]TA-9023-INF[/GOV_ID]."
        ),
        "notes": (
            "Org acronym (IRS) appears alongside a real GOV_ID. Tests that "
            "the model flags only the actual ID, not the agency abbreviation. "
            "Mixed in/out-of-scope items in one input."
        ),
    },
    {
        "id": "emoji_with_pii",
        "category": "format_variation",
        "annotated": (
            "📧 [EMAIL]chen.lim@example.org[/EMAIL] "
            "📞 [PHONE]+65 9123 4567[/PHONE]"
        ),
        "notes": (
            "Emoji-laden contact card. Each emoji is multiple code units in "
            "UTF-16 but a single code point in Python str. Tests that emoji "
            "don't break offset accounting and that the model still locates "
            "PII in non-prose layouts."
        ),
    },
    {
        "id": "mixed_pii_dense",
        "category": "density",
        "annotated": (
            "Subject: [PERSON]Dr. Yuki Sato[/PERSON] "
            "(DOB [DOB]1972-03-14[/DOB]) at "
            "[EMAIL]ysato@clinic.jp[/EMAIL] / "
            "[PHONE]+81-3-5555-2020[/PHONE], home "
            "[ADDRESS]2-1-1 Roppongi, Minato, Tokyo 106-0032[/ADDRESS], "
            "MRN [MEDICAL_ID]MRN-A40192[/MEDICAL_ID], "
            "NHI [GOV_ID]7634-2890[/GOV_ID]."
        ),
        "notes": (
            "Seven distinct PII types in one short text: PERSON, DOB, EMAIL, "
            "PHONE, ADDRESS, MEDICAL_ID, GOV_ID. Stress test for breadth — "
            "models that handle each type in isolation may miss some when "
            "they're packed together."
        ),
    },
    {
        "id": "company_name_not_person",
        "category": "near_miss",
        "annotated": "The Microsoft Corporation board approved the acquisition last week.",
        "notes": (
            "Company name that contains capitalised words but is NOT a "
            "person. Expected: spans = []. Tests that the model doesn't "
            "flag organisations as PERSON purely on capitalisation."
        ),
    },
]


def all_cases() -> list[dict]:
    return BATCH_1 + BATCH_2 + BATCH_3 + BATCH_4 + BATCH_5


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cases = all_cases()
    if not cases:
        print("No cases defined yet.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for case in cases:
        text, spans = parse_annotation(case["annotated"])
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "text": text,
            "spans": spans,
            "notes": case.get("notes", ""),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} adversarial cases to {OUT}")
    cats: dict[str, int] = {}
    for row in rows:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    for cat in sorted(cats):
        print(f"  {cat:25} {cats[cat]}")

    print("\nSpan validation: all cases parsed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
