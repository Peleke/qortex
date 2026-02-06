#!/bin/bash
# Batch ingestion of Software Design for Python Programmers chapters
#
# COST ESTIMATE: ~$10-15 for 15 chapters (~60KB each)
# RUNTIME: ~30-45 minutes total
#
# This script will:
# 1. Ingest chapters 1-4 and 6-16 (ch05 already done as implementation_hiding)
# 2. Save manifests for each chapter (recovery if anything fails)
# 3. Use auto-suggested domain names
#
# Run with: bash scripts/batch-ingest-chapters.sh
# Or for dry run: DRY_RUN=1 bash scripts/batch-ingest-chapters.sh

set -e

BOOK_DIR="/Users/peleke/Documents/Projects/qortex/data/books"
MANIFEST_DIR="/Users/peleke/Documents/Projects/qortex-track-c/data/manifests"
DRY_RUN_FLAG=""

if [ "$DRY_RUN" = "1" ]; then
    DRY_RUN_FLAG="--dry-run"
    echo "=== DRY RUN MODE - No actual extraction ==="
fi

# Create manifest directory
mkdir -p "$MANIFEST_DIR"

# Chapter list (excluding ch05 which is already ingested)
CHAPTERS=(
    "ch01_1_The_path_to_well-designed_software.txt"
    "ch02_2_Iterate_to_achieve_good_design.txt"
    "ch03_3_Get_requirements_to_build_the_right_application.txt"
    "ch04_4_Good_class_design_to_build_the_application_right.txt"
    # ch05 SKIP - already ingested as implementation_hiding
    "ch06_6_Dont_surprise_your_users.txt"
    "ch07_7_Design_subclasses_right.txt"
    "ch08_8_The_Template_Method_and_Strategy_Design_Patterns.txt"
    "ch09_9_The_Factory_Method_and_Abstract_Factory_Design_Patterns.txt"
    "ch10_10_The_Adapter_and_Façade_Design_Patterns.txt"
    "ch11_11_The_Iterator_and_Visitor_Design_Patterns.txt"
    "ch12_12_The_Observer_Design_Pattern.txt"
    "ch13_13_The_State_Design_Pattern.txt"
    "ch14_14_The_Singleton_Composite_and_Decorator_Design_Patterns.txt"
    "ch15_15_Designing_solutions_with_recursion_and_backtracking.txt"
    "ch16_16_Designing_multithreaded_programs.txt"
)

echo "=== Batch Chapter Ingestion ==="
echo "Chapters to process: ${#CHAPTERS[@]}"
echo "Book directory: $BOOK_DIR"
echo "Manifest output: $MANIFEST_DIR"
echo ""

for chapter in "${CHAPTERS[@]}"; do
    ch_num=$(echo "$chapter" | grep -oE '^ch[0-9]+' | sed 's/ch//')
    manifest_name="ch${ch_num}.manifest.json"

    echo "----------------------------------------"
    echo "Processing: $chapter"
    echo "Manifest: $MANIFEST_DIR/$manifest_name"
    echo ""

    # Run ingestion with manifest saving
    # Using yes to auto-confirm the cost prompt
    if [ -z "$DRY_RUN_FLAG" ]; then
        yes | qortex ingest "$BOOK_DIR/$chapter" \
            --save-manifest "$MANIFEST_DIR/$manifest_name" \
            $DRY_RUN_FLAG || {
            echo "WARNING: Chapter $chapter failed, continuing..."
            continue
        }
    else
        qortex ingest "$BOOK_DIR/$chapter" \
            --save-manifest "$MANIFEST_DIR/$manifest_name" \
            $DRY_RUN_FLAG
    fi

    echo ""
    echo "Completed: $chapter"
    echo ""

    # Small pause between chapters to avoid rate limiting
    sleep 2
done

echo "=== Batch Ingestion Complete ==="
echo ""
echo "View all domains: qortex inspect domains"
echo "View all rules: qortex inspect rules"
echo "Visualize: qortex viz open"
