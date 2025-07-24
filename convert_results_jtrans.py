import tqdm
import pickle
import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embedding_path):
    """
    list of {'func_name' : fun,
                'proj' : tmux,
                'ebd' : tensor }

    transform and output:
    { proj: {fun: embd}}
    """
    with open(embedding_path, 'rb') as f:
        emb_data = pickle.load(f)
    emb_dict = dict()
    for fun_dict in emb_data:
        key_bin = f"IDBs/Dataset-Muaz/{fun_dict['proj'].replace('_extract','')}.i64"
        key_fun = fun_dict['funcname']
        if key_bin in emb_dict.keys():
            emb_dict[key_bin][key_fun] = fun_dict['ebd']
        else:
            emb_dict[key_bin] = dict()
            emb_dict[key_bin][key_fun] = fun_dict['ebd']
    return emb_dict

def compute_similarity(embedding1, embedding2):
    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return float(sim[0][0])

def main(args):
    # Load metadata and embeddings
    print("[*] Loading embeddings...")
    embeddings = load_embeddings(args.embeddings)

    seen = set()
    for binary in list(embeddings.keys()):
        if "+" not in binary and binary not in seen:
            seen.add(binary)

    # Load the pairs
    print("[*] Loading function pairs...")
    pairs_df = pd.read_csv(args.pairs_csv)

    results = []

    print("[*] Computing similarities...")
    for _, row in tqdm.tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        key1 = (row['idb_path_1'], row['func_name_1'])
        key2 = (row['idb_path_2'], row['func_name_2'])

        try:
            emb1 = embeddings[key1[0]][key1[1]]
        except KeyError:
            print(f"[!] Could not find key1: {key1}")
            continue
        try:
            emb2 = embeddings[key2[0]][key2[1]]
        except KeyError:
            print(f"[!] Could not find key2: {key2}")
            continue

        sim = compute_similarity(emb1, emb2)

        results.append({
            "idb_path_1": row['idb_path_1'],
            "func_name_1": row['func_name_1'],
            "idb_path_2": row['idb_path_2'],
            "func_name_2": row['func_name_2'],
            "sim": sim
        })

    print(f"Pairs successes: {len(results)} / {len(pairs_df)}")
    results_df = pd.DataFrame(results)
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"[*] Results saved to {args.output}")
    else:
        print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", default="./pairs_testing_Dataset-Muaz.csv", help="CSV file with function pairs")
    parser.add_argument("--embeddings", default="./embeddings/Dataset-Muaz.pkl", help="Path to .pt file containing embeddings")
    parser.add_argument("--output", default="./pairs_results_Dataset-Muaz_jtr.csv", help="Path to save output CSV with similarity scores")
    args = parser.parse_args()
    main(args)
