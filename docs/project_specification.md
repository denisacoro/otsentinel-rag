# OTSentinel AI — Project Specification

## Problem statement

OTSentinel AI is a multilingual technical assistant for CPS, SCADA, Operational Technology and
Industrial IoT security. It answers English and Romanian questions using evidence retrieved from
authoritative documentation (NIST, CISA, MITRE ATT&CK for ICS, MQTT, FreeRTOS), returns exact
citations back to source passages, and explicitly refuses to answer when the corpus does not
contain sufficient evidence.

## Intended users

- OT security analysts
- Automation engineers
- Embedded developers
- CPS/security students
- Incident responders

## Supported question types

- Concept explanation ("What is network segmentation in an OT environment?")
- Advisory summary ("Summarize the impact of ICSA-XX-XXX-XX")
- Affected-product lookup ("Which Siemens products are affected by CVE-XXXX-XXXXX?")
- Mitigation recommendation ("How should MQTT broker authentication be configured?")
- ATT&CK mapping ("Which ATT&CK for ICS techniques involve Modbus abuse?")
- MQTT / FreeRTOS security explanation

## Supported languages

English and Romanian. Romanian questions may be answered from English-language source
documents (cross-lingual retrieval).

## Unsupported behaviour (explicit refusal rules)

The system must refuse or flag, rather than answer confidently, when:

- The claim would require inventing a fact not present in retrieved evidence (CVE IDs,
  version numbers, mitigation steps).
- The retrieved context does not sufficiently support an answer.
- The request asks for offensive exploitation instructions rather than defensive guidance.

## Answer contract

Every answer must include: a concise, evidence-based response; inline citations to source
chunk IDs; an explicit uncertainty statement when evidence is partial; and the response in the
requested language.

## Success criteria

- **Retrieval:** Recall@10 and MRR@10 measured against a golden evaluation set (not yet built).
- **Generation:** Faithfulness and answer correctness via RAGAS, once generation exists.
- **Citations:** Every claim in an answer resolves to a real, retrievable chunk ID.
- **Latency:** P95 end-to-end query latency reported once the API exists.
- **Deployment:** Full stack (API, vector DB, model server) starts from one documented
  `docker compose up` command.

## Current implementation status

See `README.md` for what is built versus outstanding. As of this document, ingestion,
parsing and dense indexing exist for NIST SP 800-82r3 and MQTT 5.0. Retrieval beyond a CLI
search command, generation, evaluation and fine-tuning have not been started.
