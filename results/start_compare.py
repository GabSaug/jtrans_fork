import argparse
import pickle
from tqdm import tqdm

import torch


def log(infostr, LOG):
    #print(infostr)
    LOG.write(infostr)


def preprocessJtransOutput(embeddings_list):
    """
    Transform: [ emb_dict ]
    embd_dict:{ proj : "sqlite3", funcname : "foo", O2 : torch.Tensor }
    ->
    { "sqlite3" : { foo1 : torch.tensor,
                    foo2 : torch.tensor, ...}
                    }
    NOTE: Is it expected (and will be checked) that each binary only has one
    obfuscation level
    """

    output_dict = dict()
    for emb_dict in embeddings_list:
        if len(emb_dict.keys()) != 3:
            print("Error, embedding entry has more than one opt level:{}".format(
                emb_dict.keys()))
        binary, func, torchEmb = emb_dict.values()
        if binary not in output_dict.keys():
            output_dict[binary] = {func: torchEmb}
        else:
            output_dict[binary].update({func: torchEmb})
    return output_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="jTrans-Compare_Binaries")
    parser.add_argument("csv_path", type=str, help="Path to the function pairs CSV file ")
    parser.add_argument("in_pickle_path", type=str, help="Path to the embeddings pickle file")
    parser.add_argument("--output", type=str, default="./output.csv", help="Path to output csv file")
    parser.add_argument("--log", type=str, default="./stat_compare_log.txt", help="Path to the log file")
    args = parser.parse_args()

    csv_in = open(args.csv_path, "r")
    csv_lines = csv_in.readlines()
    csv_lines = csv_lines[1:]

    pickle_in = open(args.in_pickle_path, "rb")
    embeddings_list = pickle.load(pickle_in)
    pickle_in.close()

    csv_out = open(args.output, "w")
    csv_out.write("idb_path_1,func_name_1,idb_path_2,func_name_2,sim\n")

    LOG = open(args.log, "w")

    embeddings = preprocessJtransOutput(embeddings_list)
    breakpoint()

    #  Pour chaque ligne du csv, on cherche l'embedding des 2 fonctions dans le .pkl et calcule leur cosinus
    for pair in tqdm(csv_lines, desc="Parsing csv pairs..."):
        pair = pair.replace("IDBs/Dataset-Muaz/", "").replace(".i64", "").replace("\n", "").replace("-2", "")
        bin_name_1, func_name_1, bin_name_2, func_name_2 = pair.split(",")

        embedding_f_1 = None
        embedding_f_2 = None

        if bin_name_1 not in embeddings.keys():
            log("[*] Error, binary {} not in embeddings".format(bin_name_1), LOG)
            continue

        if bin_name_2 not in embeddings.keys():
            log("[*] Error, binary {} not in embeddings".format(bin_name_2), LOG)
            continue

        if func_name_1 not in embeddings[bin_name_1].keys():
            log("[*] Error, func {} not in {} embeddings".format(
                func_name_1, bin_name_1), LOG)
            continue

        if func_name_2 not in embeddings[bin_name_2].keys():
            log("[*] Error, func {} not in {} embeddings".format(
                func_name_2, bin_name_2), LOG)
            continue

        emb1 = embeddings[bin_name_1][func_name_1]
        emb2 = embeddings[bin_name_2][func_name_2]

        # On calcule leur cosine similarity
        log("Match found for the line: {}\n".format(pair), LOG)
        cosi = torch.nn.CosineSimilarity(dim=1)
        csv_out.write("{},{},{},{},{}\n".format(
            bin_name_1, func_name_1, bin_name_2, func_name_2,
            float(cosi(emb1, emb2))))

    csv_in.close()
    csv_out.close()
    LOG.close()
