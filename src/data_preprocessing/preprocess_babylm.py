import argparse
import os
import sys
from pathlib import Path
import spacy
from spacy.cli import download
from tqdm import tqdm
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def ensure_spacy_model(model_name="en_core_web_sm"):
    """Ensure the required spaCy model is downloaded"""
    try:
        nlp = spacy.load(
            model_name,
            disable=[
                "tagger",
                "parser",
                "ner",
                "lemmatizer",
                "attribute_ruler",
                "tokenizer",
            ],
        )
        nlp.enable_pipe("senter")
    except OSError:
        print(f"Downloading spacy model {model_name}...")
        download(model_name)
        nlp = spacy.load(
            model_name,
            disable=[
                "tagger",
                "parser",
                "ner",
                "lemmatizer",
                "attribute_ruler",
                "tokenizer",
            ],
        )
        nlp.enable_pipe("senter")
    return nlp


def preprocess_childes_line(line):
    """Preprocess a line from CHILDES format
    - Removes speaker markers (*XXX:)
    - Removes content in square brackets [...]
    - Returns None if the line should be skipped
    """
    line = line.strip()

    # Skip empty lines
    if not line:
        return None

    # Skip lines starting with = (file headers)
    if line.startswith("="):
        return None

    # Skip lines starting with % (annotations)
    if line.startswith("%"):
        return None

    # Remove speaker markers (*XXX:)
    if line.startswith("*"):
        line = line.split(":", 1)[1] if ":" in line else ""

    # Remove content in square brackets, including [shakes head "no"]
    while "[" in line and "]" in line:
        start = line.find("[")
        end = line.find("]", start) + 1
        if end == 0:  # No closing bracket found
            break
        line = line[:start] + line[end:]

    # Skip if line is empty after preprocessing
    line = line.strip()
    if not line:
        return None

    return line


def preprocess_switchboard_line(line):
    """Preprocess a line from Switchboard format
    - Removes speaker markers (A:, B:, etc.)
    - Returns None if the line should be skipped
    """
    line = line.strip()

    # Skip empty lines
    if not line:
        return None

    # Remove speaker markers (A:, B:, etc.)
    if line and line[0].isalpha() and len(line) > 1 and line[1] == ":":
        line = line[2:].strip()

    # Skip if line is empty after preprocessing
    if not line:
        return None

    return line


def process_text_file(input_file, output_file, nlp, min_length=5):
    """Process a single text file using the provided spaCy model"""
    print(f"Processing {input_file}...")

    input_path = str(input_file).lower()
    is_childes = "childes" in input_path
    is_switchboard = "switchboard" in input_path

    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if is_childes:
                line = preprocess_childes_line(line)
            elif is_switchboard:
                line = preprocess_switchboard_line(line)

            if line is None or not line:  # Skip empty lines or None
                continue

            doc = nlp(line.strip())
            for sent in doc.sents:
                tokens = [token.text for token in sent if not token.is_space]
                if tokens and len(tokens) >= min_length:
                    f_out.write(" ".join(tokens) + "\n")


def process_single_file(model_name, min_length, input_file):
    """Process a single file with its own spaCy model instance"""
    nlp = ensure_spacy_model(model_name)

    # Create a temporary file with a unique name
    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(temp_fd)

    process_text_file(input_file, temp_path, nlp, min_length)
    return temp_path


def process_directory(
    input_dir, output_path, model_name="en_core_web_sm", min_length=2, n_jobs=None
):
    """Process all text files in the input directory and concatenate results"""
    input_files = sorted([f for f in Path(input_dir).glob("*") if f.is_file()])

    # Create a process pool and process files in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        process_func = partial(process_single_file, model_name, min_length)
        temp_files = list(
            tqdm(
                executor.map(process_func, input_files),
                total=len(input_files),
                desc="Processing files",
            )
        )

    # Concatenate all processed files
    print("Concatenating files...")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for temp_file in tqdm(temp_files, desc="Concatenating"):
            with open(temp_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            os.unlink(temp_file)  # Delete temporary file


def main():
    parser = argparse.ArgumentParser(
        description="Process text files and output tokenized sentences"
    )
    parser.add_argument("--input_dir", help="Input directory containing text files")
    parser.add_argument("--output_path", help="Output text file path")
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum number of tokens required for a sentence (default: 2)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: all available cores)",
    )

    args = parser.parse_args()

    try:
        # Verify input directory exists
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Process all files in the directory
        process_directory(
            args.input_dir,
            args.output_path,
            args.spacy_model,
            args.min_length,
            args.n_jobs,
        )

        print(f"Output written to: {args.output_path}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
