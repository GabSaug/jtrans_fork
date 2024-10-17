import argparse
import pickle

import torch

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="jTrans-Compare_Binaries")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file input ")
    parser.add_argument("--in_pickle_path", type=str, help="Path to the input pickle file")
    parser.add_argument("--output", type=str, default="./output.csv", help="Path to the input pickle file")

    args = parser.parse_args()

    csv_in = open(args.csv_path, "r")
    csv_lines = csv_in.readlines()

    pickle_in = open(args.in_pickle_path, "rb")
    embeddings_list = pickle.load(pickle_in)
    pickle_in.close()

    csv_out = open(args.output_csv, "w")
    csv_out.write("index,,idb_path_1,fva_1,func_name_1,idb_path_2,fva_2,func_name_2,db_type,sim")
    
    #  Pour chaque ligne du csv, on cherche l'embedding des 2 fonctions dans le .pkl et calcule leur cosinus 
    for pair in csv_lines:
        idb_path_1, func_name_1, idb_path_2, func_name_2 = pair.split(",")

        bin_name_1 == idb_path1.split("/")[-1]
        bin_name_2 == idb_path2.split("/")[-1]
        
        embedding_f_1 = None
        embedding_f_2 = None
        
        for i in range(len(embedding_list)):
            if (embedding_f_1 != None) and (embeddings_list[i]["proj"] == bin_name_1) :
                if embeddings_list[i]["funcname"] == func_name_1:
                    embeddings_f_1 = embeddings_list[i]["Op"]

            if (embedding_f_2 != None) and (embeddings_list[i]["proj"] == bin_name_2) :
                if embeddings_list[i]["funcname"] == func_name_2:
                    embeddings_f_2 = embeddings_list[i]["Op"]
                    

            #Si on a trouve les 2 fonctions dans le fichier pickle
            if embedding_f_1 != None and embedding_f_2 != None :
                break

        
        if embedding_f_1 != None and embedding_f_2 != None : 
            #On calcule leur cosine similarity
            cosi = torch.nn.CosineSimilarity(dim=1)
            csv_out.write(pair + "," + cosi(embedding_f_1, embedding_f_2))
            
        else:
            print("NO MATCH FOUND FOR THE LINE :" + pair)

        
        

    csv_in.close()
    csv_out.close()
    

