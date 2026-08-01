# OTSentinel AI — Golden Evaluation Dataset Card

## Version
v1 — <eac950b4636558a6d660d63f9ef5fe3f2bad4c1aa2be44fc37c206ec093571f4>

## Scope
32 manually authored questions grounded in the two sources currently indexed:
NIST SP 800-82r3 (14 questions) and MQTT 5.0 (12 questions), plus 6 deliberately
unanswerable questions covering advisory/CVE, ATT&CK, FreeRTOS and multi-document
gaps in the current corpus.

This is intentionally smaller than the 200-300 question target in
docs/project_specification.md. The corpus only covers 2 of the 5 planned source
families; writing hundreds of questions now would mean padding with repetitive
content or questions the system cannot honestly answer yet. This dataset will
grow incrementally as CISA advisories, MITRE ATT&CK for ICS, and FreeRTOS
documentation are ingested.

## Composition
- 26 answerable, 6 unanswerable (18.75%)
- 29 English, 3 Romanian
- Splits: 23 validation, 9 test (grouped by topic cluster, not randomly shuffled)

## Methodology
Questions were written against chunks sampled evenly across each indexed
document (scripts/sample_eval_candidates.py), then manually verified against
the real chunk text before being included. Every relevant_chunk_id is a real,
existing chunk ID; no chunk ID or reference answer was invented.

## Known limitations
- advisory_summary, affected_product_lookup and attack_mapping question types
  currently have no answerable examples, since CISA and MITRE ATT&CK are not
  yet ingested. Placeholder unanswerable questions of these types exist and
  should be relabeled once those sources are added.
- geval-032 documents a known baseline retrieval limitation (plain dense
  top-5 search doesn't reliably surface a second document's evidence in a
  blended query) rather than a corpus gap. Revisit after step 10.

## Usage rules
- The `test` split must not be used to tune prompts, chunking, or retrieval
  configuration. Use only `validation` questions during step 10 experiments.
- Do not use any question from this file as a fine-tuning example in step 11.