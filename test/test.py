import os
import sys
import pr
import se
import sb
import utils.PRASEUtils as pu
from time import strftime, localtime
import argparse

parser = argparse.ArgumentParser(description="Parameters for the script.")
parser.add_argument('--dataset', default="KKS", help='the dataset of the experiment')
parser.add_argument('--candidate_num', default=1, type=int, help='the top-k prediction for each entity')
parser.add_argument('--sbert_eqv_weight', default=0.1, type=float, help='the weight for sbert equivalent probability')
parser.add_argument('--emb_eqv_weight', default=0.1, type=float, help='the weight for SE embedding equivalent probability')
parser.add_argument('--ent_eqv_thre', default=0, type=float, help='the threshold for final prediction')
parser.add_argument('--PR_iteration_num', default=10, type=int, help='the interation num in PR module')
parser.add_argument('--normalize_pow', default=-1, type=float, help='the power of normalizarion in PR module')
args = parser.parse_args()

def get_time_str():
    return str(strftime("[%Y-%m-%d %H:%M:%S]: ", localtime()))


def print_kgs_stat(kgs_obj):
    print(get_time_str() + "Discovered entity mapping number: " + str(len(kgs_obj.get_ent_align_ids_result())))
    sys.stdout.flush()
    rel_align_a, rel_align_b = kgs_obj.get_rel_align_ids_result()
    print(get_time_str() + "Discovered relation mapping number: " + str(len(rel_align_a) + len(rel_align_b)))
    sys.stdout.flush()
    attr_align_a, attr_align_b = kgs_obj.get_attr_align_name_result()
    print(get_time_str() + "Discovered attribute mapping number: " + str(len(attr_align_a) + len(attr_align_b)))
    sys.stdout.flush()


def save_alignment(kgs_obj, iter, module):
    # output_dir = f"output/KKdata_V4/last_multi/ent_cand_3"
    # output_dir = f"output/KKdata_V4/only_labeled_vod/sbert_weight_{SBERT_EQV_WEIGHT}/set_ent_candidate_{CANDIDATE_NUM}"
    # output_dir = f"output/KKdata_V4/labeled_vod_tv/sbert_weight_{SBERT_EQV_WEIGHT}/set_ent_candidate_{CANDIDATE_NUM}"
    # output_dir = f"output/KKdata_V4/sbert_weight/sbert_weight_{SBERT_EQV_WEIGHT}/set_ent_candidate_{CANDIDATE_NUM}"
    # output_dir = f"output/normalize_test/normalize_pow_{NORMALIZE_POW}/sbert_weight_{SBERT_EQV_WEIGHT}"
    output_dir = f"output/{args.dataset}/cand_{args.candidate_num}_sb_{args.sbert_eqv_weight}_norm_{args.normalize_pow}/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{module}_iter_{iter}.txt")
    with open(output_path, 'w') as fout:
        for tuple in kgs_obj.get_ent_align_name_result():
            if tuple[0] != "" and tuple[1] != "" and tuple[0] in kg1_ent_set:
                fout.write(tuple[0]+'\t'+tuple[1]+'\t'+str(tuple[2])+'\n')

def load_entity(rel_path_1, rel_path_2):
    kg1_ent_set, kg2_ent_set = set(), set()
    with open(rel_path_1, "r") as f:
        data = f.readlines()
        for row in data:
            row = row.strip().split('\t')
            kg1_ent_set.add(row[0])
    with open(rel_path_2, "r") as f:
        data = f.readlines()
        for row in data:
            row = row.strip().split('\t')
            kg2_ent_set.add(row[0])
    return kg1_ent_set, kg2_ent_set

def print_kg_stat(kg_obj):
    print(get_time_str() + "Entity Number: " + str(len(kg_obj.get_ent_id_set())))
    print(get_time_str() + "Relation Number: " + str(int(len(kg_obj.get_rel_id_set()) / 2)))
    print(get_time_str() + "Attribute Number: " + str(int(len(kg_obj.get_attr_id_set()) / 2)))
    print(get_time_str() + "Literal Number: " + str(int(len(kg_obj.get_lite_id_set()) / 2)))
    print(get_time_str() + "Relation Triple Number: " + str(int(len(kg_obj.get_relation_id_triples()) / 2)))
    print(get_time_str() + "Attribute Triple Number: " + str(int(len(kg_obj.get_attribute_id_triples()) / 2)))
    sys.stdout.flush()

path = os.path.abspath(__file__)
base, _ = os.path.split(path)

output_dir = os.path.join(base, f"results/{args.dataset}")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"cand_{args.candidate_num}_sb_{args.sbert_eqv_weight}_norm_{args.normalize_pow}_results")
f = open(output_path, "w")
f.close()

### Dataset
if args.dataset == "dbp_wd_15k_V1":
    dataset_dir = "/tmp2/yhwang/EA_dataset/DWY15K/dbp_wd_15k_V1"
    kg1_rel_path = os.path.join(dataset_dir, "rel_triples_1")
    kg1_attr_path = os.path.join(dataset_dir, "attr_triples_1")
    kg2_rel_path = os.path.join(dataset_dir, "rel_triples_2")
    kg2_attr_path = os.path.join(dataset_dir, "attr_triples_2")
    test_path = os.path.join(dataset_dir, "ent_links")
    
elif args.dataset == "dbp_yg_15k_V1":
    dataset_dir = "/tmp2/yhwang/EA_dataset/DWY15K/dbp_yg_15k_V1"
    kg1_rel_path = os.path.join(dataset_dir, "rel_triples_1")
    kg1_attr_path = os.path.join(dataset_dir, "attr_triples_1")
    kg2_rel_path = os.path.join(dataset_dir, "rel_triples_2")
    kg2_attr_path = os.path.join(dataset_dir, "attr_triples_2")
    test_path = os.path.join(dataset_dir, "ent_links")
    
elif args.dataset == "KKS":
    dataset_dir = "/tmp2/yhwang/EA_dataset/KKdata_V4"
    kg1_rel_path = os.path.join(dataset_dir, "vod_triplet.txt")
    kg1_attr_path = os.path.join(dataset_dir, "vod_triplet.txt")
    kg2_rel_path = os.path.join(dataset_dir, "tv_triplet.txt")
    kg2_attr_path = os.path.join(dataset_dir, "tv_triplet.txt")
    test_path = os.path.join(dataset_dir, "ent_mapping.txt")

else:
    raise BaseException("Invalid dataset!")

# dataset_dir = "/tmp2/yhwang/EA_dataset/DWY100K/DWY100K_raw_data/dbp_wd"
# dataset_dir = "/tmp2/yhwang/EA_dataset/DWY100K/DWY100K_raw_data/dbp_yg"
# dataset_dir = "/tmp2/yhwang/EA_dataset/SRPRS/dbp_wd_15k_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/SRPRS/dbp_yg_15k_V2"

### OpenEA Dataset v1.1
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_W_15K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_Y_15K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_W_100K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_Y_100K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_W_15K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_Y_15K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_W_100K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v1.1/D_Y_100K_V2"

### OpenEA Dataset v2.0
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_W_15K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_Y_15K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_W_100K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_Y_100K_V1"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_W_15K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_Y_15K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_W_100K_V2"
# dataset_dir = "/tmp2/yhwang/EA_dataset/OpenEA_dataset_v2.0/D_Y_100K_V2"

print(get_time_str() + f"Use Dataset {args.dataset}")
sys.stdout.flush()

kg1_ent_set, kg2_ent_set = load_entity(kg1_rel_path, kg2_rel_path)

print(get_time_str() + "Sentence Bert Inferencing...")
sys.stdout.flush()


### Use preload embedding to reduce experiment time
if args.dataset == "KKS":
    import pickle
    with open("/tmp2/yhwang/EA_dataset/KKdata_V4/vod_sbert_dict.pkl", "rb") as f:
        kg1_sbert_dict = pickle.load(f)
    with open("/tmp2/yhwang/EA_dataset/KKdata_V4/tv_sbert_dict.pkl", "rb") as f:
        kg2_sbert_dict = pickle.load(f)
###

else:
    SBert = sb.SBert(args.dataset)
    kg1_sbert_dict, kg2_sbert_dict = SBert.get_sbert_dict(kg1_ent_set, kg2_ent_set)

print(get_time_str() + "Construct source KG...")
sys.stdout.flush()

# construct source KG from file
kg1 = pu.construct_kg(kg1_rel_path, kg1_attr_path)
print_kg_stat(kg1)
for key in kg1_sbert_dict.keys():
    kg1.insert_ent_sbert_embed_by_name(key, kg1_sbert_dict[key])

print(get_time_str() + "Construct target KG...")
sys.stdout.flush()

# construct target KG from file
kg2 = pu.construct_kg(kg2_rel_path, kg2_attr_path)
print_kg_stat(kg2)
for key in kg2_sbert_dict.keys():
    kg2.insert_ent_sbert_embed_by_name(key, kg2_sbert_dict[key])

print(get_time_str() + "Construct KGs...")
sys.stdout.flush()

# construct KGs object
kgs = pu.construct_kgs(kg1, kg2)

# configure kgs
kgs.set_se_module(se.GCNAlign)
kgs.set_pr_module(pr.PARIS)

# Set Thread Number:
# kgs.pr.set_worker_num(6)  # default is the hardware concurrency of the thread

# Set Entity Candidate Number:
CANDIDATE_NUM = args.candidate_num
kgs.pr.set_ent_candidate_num(CANDIDATE_NUM)
print(get_time_str() + "Entity Candidate Number: " + str(CANDIDATE_NUM))

# Set SBert Eqvalence Weight:
SBERT_EQV_WEIGHT = args.sbert_eqv_weight
kgs.pr.set_sbert_eqv_weight(SBERT_EQV_WEIGHT)
print(get_time_str() + "SBert Eqvalence Weight: " + str(SBERT_EQV_WEIGHT))

# Set Embedding Eqvalence Weight:
EMB_EQV_WEIGHT = args.emb_eqv_weight
kgs.pr.set_emb_eqv_weight(EMB_EQV_WEIGHT)
print(get_time_str() + "Embedding Eqvalence Weight: " + str(EMB_EQV_WEIGHT))

ENT_EQV_THRE = args.ent_eqv_thre
kgs.pr.set_ent_eqv_bar(ENT_EQV_THRE)
print(get_time_str() + "Entity Eqvalence Threshold: " + str(ENT_EQV_THRE))

PR_ITERATION_NUM = args.PR_iteration_num
kgs.pr.set_max_iteration_num(PR_ITERATION_NUM)
print(get_time_str() + "PR Iteration Number: " + str(PR_ITERATION_NUM))

NORMALIZE_POW = args.normalize_pow
kgs.pr.set_normalize_pow(NORMALIZE_POW)
print(get_time_str() + "PR Normalize Power: " + str(NORMALIZE_POW))

'''
Load PRASEMap Model:
pu.load_prase_model(kgs, load_path)
'''

# init kgs
kgs.init()

print(get_time_str() + "Performing PR Module (PARIS)...")
sys.stdout.flush()

# run pr module
kgs.run_pr()
save_alignment(kgs, 0, "PR")
print_kgs_stat(kgs)
kgs.test(test_path, output_path, threshold=[0.1 * i for i in range(10)], iter="PR (iter = 0) : ")

kgs.pr.enable_rel_init(False)

iteration = 1
for i in range(iteration):
    print(get_time_str() + "Performing SE Module (GCNAlign)...")
    sys.stdout.flush()
    # run se module
    kgs.run_se(embedding_feedback=True, mapping_feedback=True)
    save_alignment(kgs, i+1, "SE")

    print_kgs_stat(kgs)
    kgs.test(test_path, output_path, threshold=[0.1 * i for i in range(10)], iter=f"SE (iter = {i+1}) : ")
    print(get_time_str() + "Performing PR Module (PARIS)...")
    sys.stdout.flush()
    # run pr module
    # if i == iteration-1:
    #     kgs.pr.set_ent_candidate_num(50)
    kgs.run_pr()
    save_alignment(kgs, i+1, "PR")

    print_kgs_stat(kgs)
    kgs.test(test_path, output_path, threshold=[0.1 * i for i in range(10)], iter=f"PR (iter = {i+1}) : ")

# Save PRASE Model:
# save_path = os.path.join(base, "models/KKdata_V4/PRASE_model")
# pu.save_prase_model(kgs, save_path)
