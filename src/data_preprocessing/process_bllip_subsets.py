import os
import re
from pathlib import Path
from nltk import Tree
from typing import List, Dict
from tqdm import tqdm

# Define the section numbers for each subset
SECTIONS = {
    "MD": {
        "1987": [
            5,
            10,
            18,
            21,
            22,
            26,
            32,
            35,
            43,
            47,
            48,
            49,
            51,
            54,
            55,
            56,
            57,
            61,
            62,
            65,
            71,
            77,
            79,
            81,
            90,
            96,
            100,
            105,
            122,
            125,
        ],
        "1988": [
            12,
            13,
            14,
            17,
            23,
            24,
            33,
            39,
            40,
            47,
            48,
            54,
            55,
            59,
            69,
            72,
            73,
            76,
            78,
            79,
            83,
            84,
            88,
            89,
            90,
            93,
            94,
            96,
            102,
            107,
        ],
        "1989": list(range(12, 42)),  # 012-041
    },
    "SM": {
        "1987": [35, 43, 48, 54, 61, 71, 77, 81, 96, 122],
        "1988": [24, 54, 55, 59, 69, 73, 76, 79, 90, 107],
        "1989": [12, 13, 15, 18, 21, 22, 28, 37, 38, 39],
    },
    "XS": {"1987": [71, 122], "1988": [54, 107], "1989": [28, 37]},
}


def remove_tree_annotations(tree_str: str) -> str:
    """Remove tree annotations and extract just the words, excluding those under -NONE- labels."""
    try:
        tree = Tree.fromstring(tree_str)
        # Get words but skip those whose parent has label -NONE-
        words = []
        for pos in tree.treepositions("leaves"):
            parent_pos = pos[:-1]  # Get position of parent
            if tree[parent_pos].label() != "-NONE-":
                words.append(tree[pos])
        return " ".join(words)
    except Exception as e:
        print(f"Error parsing tree: {e}")
        return ""


def process_file(file_path: str) -> List[str]:
    """Process a single file and return list of cleaned sentences."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Split by S1 to get individual sentences
    sentences = re.split(r"\n\(S1", content)
    # Clean up and process each sentence
    cleaned_sentences = []
    for sent in sentences:
        if not sent.strip():
            continue
        # Add back the S1 if it was removed by split
        if not sent.startswith("(S1"):
            sent = "(S1" + sent
        cleaned = remove_tree_annotations(sent.strip())
        if cleaned:
            cleaned_sentences.append(cleaned)

    return cleaned_sentences


def main():
    # Base paths
    input_base = Path("data/bliip_87_89_wsj")
    output_base = Path("data/BLLIP")

    # Process each subset size
    for size, year_sections in SECTIONS.items():
        print(f"\nProcessing {size} subset...")
        all_sentences = []

        # Process each year
        for year, sections in tqdm(year_sections.items(), desc="Years"):
            year_path = input_base / year

            # Process each section
            for section in tqdm(sections, desc=f"Sections ({year})", leave=False):
                # Format section number with leading zeros
                section_pattern = f"w*_{section:03d}"

                # Find matching section directories
                matching_sections = list(year_path.glob(section_pattern))
                if not matching_sections:
                    print(
                        f"Warning: No matching section found for pattern: {year_path}/{section_pattern}"
                    )
                    continue

                # Process all files in matching sections
                for section_path in matching_sections:
                    for file_path in section_path.glob("*"):
                        if file_path.is_file():
                            sentences = process_file(str(file_path))
                            all_sentences.extend(sentences)

        # Create output directory if it doesn't exist
        output_dir = output_base / size
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write all sentences to output file
        output_file = output_dir / "train.txt"
        print(f"Writing {len(all_sentences)} sentences to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in tqdm(all_sentences, desc="Writing sentences"):
                f.write(sentence + "\n")

        print(f"Completed {size} subset: {len(all_sentences)} sentences")


if __name__ == "__main__":
    main()
