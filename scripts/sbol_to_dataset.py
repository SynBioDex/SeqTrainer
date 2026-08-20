from sbol2 import Document
import pandas as pd
import argparse


def sbol_to_csv(input_file, output_file="dataset.csv"):
    doc = Document()
    doc.read(input_file)

    data = []

    for comp in doc.componentDefinitions:
        seq = None

        try:
            if comp.sequences:
                seq_id = comp.sequences[0]
                sequence_obj = doc.sequences.get(seq_id)

                if sequence_obj:
                    seq = sequence_obj.elements
        except Exception:
            continue

        if seq:
            data.append({
                "sequence": seq,
                "label": "unknown"
            })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SBOL to CSV dataset")
    parser.add_argument("--input", required=True, help="Input SBOL file")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV file")

    args = parser.parse_args()

    sbol_to_csv(args.input, args.output)