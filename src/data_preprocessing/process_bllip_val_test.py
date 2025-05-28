import os
from pathlib import Path
from nltk import Tree
from tqdm import tqdm


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


def process_split(input_file: str, output_files: list) -> None:
    """Process a single split (val/test) and write to multiple size directories."""
    print(f"\nProcessing {input_file}...")

    # Read and process input file
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = f.readlines()

    # Process each sentence
    processed_sentences = []
    for sent in tqdm(sentences, desc="Processing sentences"):
        cleaned = remove_tree_annotations(sent.strip())
        if cleaned:
            processed_sentences.append(cleaned)

    # Write to all output files
    print(f"Writing {len(processed_sentences)} sentences to output files...")
    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in processed_sentences:
                f.write(sentence + "\n")


def main():
    # Define paths
    val_test_dir = Path("data/BLLIP_val_test")
    output_dir = Path("data/BLLIP")
    sizes = ["MD", "SM", "XS"]

    # Process validation set
    val_input = val_test_dir / "valid.txt"
    val_outputs = [output_dir / size / "dev.txt" for size in sizes]
    process_split(val_input, val_outputs)

    # Process test set
    test_input = val_test_dir / "test.txt"
    test_outputs = [output_dir / size / "test.txt" for size in sizes]
    process_split(test_input, test_outputs)


if __name__ == "__main__":
    main()
