import pandas as pd
import glob
import os

input_dir = "./results_pretrained_def"
output_file = "combined_dimtrx_pretrained_def.tsv"

files = glob.glob(os.path.join(input_dir, "*.csv"))

def clean_ids(idx):
    return idx.astype(str).str.strip().str.strip('"').str.strip("'")

matrices = []
query_ids = []      # IDs that actually appear as a row (i.e., were queried)
header_ids = set()  # IDs that appear as columns (i.e., the reference headers)

# Read files and collect IDs
for f in files:
    print("Reading:", f)
    df = pd.read_csv(f, index_col=0, sep='\t')
    df.index = clean_ids(df.index)
    df.columns = clean_ids(df.columns)

    query_ids.extend(df.index.tolist())
    header_ids.update(df.columns)

    matrices.append(df)

query_ids = sorted(set(query_ids))
# Make sure every query also has a column (for the diagonal), even if it
# wasn't part of some other file's header row
all_cols = sorted(header_ids | set(query_ids))

print("Number of query rows (files):", len(query_ids))
print("Total unique header IDs:", len(all_cols))

# Initialize final matrix: rows = queries, columns = all headers seen
combined = pd.DataFrame(
    float("nan"),
    index=query_ids,
    columns=all_cols
)

# Merge all matrices — each df fills in its own query row(s)
for df in matrices:
    aligned = df.reindex(index=query_ids, columns=all_cols)
    combined.update(aligned)

# Set diagonal (query vs itself) to 0, only where that query is also a column
for gid in query_ids:
    if gid in combined.columns:
        combined.loc[gid, gid] = 0

# Fill missing (never-observed) pairwise distances with -1
combined = combined.fillna(-1)

# Save tab-separated
combined.to_csv(output_file, sep="\t", header=True)
print("Saved:", output_file)
