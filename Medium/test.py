"""
Test script for the Data Ingestion + Quality Pipeline.

Runs the full pipeline: load documents → noise filter → dedup → PII mask → freshness.

Usage:
    python test_ingestion.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    print("=" * 60)
    print("  RAG INGESTION + QUALITY PIPELINE TEST")
    print("=" * 60)

    loader = DocumentLoader(ocr_enabled=False)
    kb_path = Path("knowledge_base/it_docs")

    print(f"\nSource: {kb_path.resolve()}\n")

    # ── Run full pipeline: load + quality checks ──
    print("-" * 60)
    print("  Running: Load -> Noise Filter -> Metadata -> Dedup -> PII -> Freshness")
    print("-" * 60)

    clean_docs, quality_report = loader.load_and_validate(
        kb_path,
        enable_dedup=True,
        enable_noise_filter=True,
        enable_pii=True,
        enable_freshness=True,
        enable_metadata=True,
        mask_pii=True,
    )

    # ── Per-document results ──
    for doc_info in quality_report["per_document"]:
        filename = doc_info["filename"]
        status = doc_info["status"]

        print(f"\n{'-' * 60}")
        print(f"  {filename}  [{status}]")
        print(f"{'-' * 60}")

        # Noise filter results
        noise = doc_info.get("noise", {})
        if noise:
            print(f"  Noise:     {noise.get('removed_chars', 0)} chars removed "
                  f"(SNR: {noise.get('signal_to_noise', 100)}%)")
            if noise.get("patterns"):
                print(f"             Patterns: {', '.join(noise['patterns'])}")

        # Dedup results
        dedup = doc_info.get("dedup", {})
        if dedup:
            dup_status = "DUPLICATE" if dedup["is_duplicate"] else "unique"
            print(f"  Dedup:     {dup_status} (method: {dedup['method']})")

        # PII results
        pii = doc_info.get("pii", {})
        if pii:
            if pii["has_pii"]:
                print(f"  PII:       DETECTED -> {pii['entities']} (masked via {pii['method']})")
            else:
                print(f"  PII:       clean")

        # Metadata validation results
        meta = doc_info.get("metadata", {})
        if meta:
            print(f"  Metadata:  completeness={meta['completeness']}%, "
                  f"consistency={meta['consistency']}%")
            if meta.get("missing_fields"):
                print(f"             Missing: {', '.join(meta['missing_fields'])}")
            if meta.get("auto_extracted"):
                extracted = ", ".join(f"{k}={v}" for k, v in meta["auto_extracted"].items())
                print(f"             Auto-extracted: {extracted}")
            if meta.get("normalized"):
                for fld, desc in meta["normalized"].items():
                    print(f"             Normalized {fld}: {desc}")

        # Freshness results
        fresh = doc_info.get("freshness", {})
        if fresh:
            print(f"  Freshness: {fresh['status']} "
                  f"(age: {fresh['age_days']}d, score: {fresh['score']})")

    # ── Overall quality metrics ──
    print(f"\n{'=' * 60}")
    print("  QUALITY METRICS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Documents In:          {quality_report['total_input']}")
    print(f"  Documents Out:         {quality_report['total_output']}")
    print(f"  Duplicates Removed:    {quality_report['duplicates_removed']}")
    print(f"  Stale Documents:       {quality_report['stale_documents']}")
    print(f"  Docs with PII:         {quality_report['pii_documents']}")
    print(f"  Avg Signal-to-Noise:   {quality_report['avg_signal_to_noise']}%")
    print(f"  Deduplication Rate:    {quality_report['dedup_rate']}%")
    print(f"  Freshness Score:       {quality_report['freshness_score']}%")
    print(f"  PII Detection Rate:    {quality_report['pii_rate']}%")
    print(f"  Metadata Completeness: {quality_report['metadata_completeness']}%")
    print(f"  Metadata Consistency:  {quality_report['metadata_consistency']}%")

    # ── Show a PII-masked doc as proof ──
    for doc in clean_docs:
        if "support_ticket" in doc.metadata.get("filename", ""):
            print(f"\n{'=' * 60}")
            print(f"  PII MASKING DEMO: {doc.metadata['filename']}")
            print(f"{'=' * 60}")
            print(doc.content)
            break

    print()


if __name__ == "__main__":
    main()