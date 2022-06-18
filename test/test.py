import torch
import csv
import os
import sys
import pr
import se
import utils.PRASEUtils as pu
from time import strftime, localtime


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

# def save_alignment(kgs_obj):
#     # output_path = f"output/v4_0.9/iter_{iteration}_ent_align.txt"
#     # output_path = f"output/KKdata_V4/iter_test/iter_{iteration}_1.txt"
#     output_path = f"output/KKdata_V4/last_PR_multi/iter_{iteration}_candidate_5_without_visited.txt"
#     with open(output_path, 'w') as fout:
#         for tuple in kgs_obj.get_ent_align_name_result():
#             fout.write(tuple[0]+'\t'+tuple[1]+'\t'+str(tuple[2])+'\n')

def save_alignment(kgs_obj, iter, module):
    output_path = f"output/KKdata_V4/sbert_weight/sbert_weight_{SBERT_EQV_WEIGHT}/{module}_iter_{iter}.txt"
    with open(output_path, 'w') as fout:
        for tuple in kgs_obj.get_ent_align_name_result():
            fout.write(tuple[0]+'\t'+tuple[1]+'\t'+str(tuple[2])+'\n')

def print_kg_stat(kg_obj):
    print(get_time_str() + "Entity Number: " + str(len(kg_obj.get_ent_id_set())))
    print(get_time_str() + "Relation Number: " + str(int(len(kg_obj.get_rel_id_set()) / 2)))
    print(get_time_str() + "Attribute Number: " + str(int(len(kg_obj.get_attr_id_set()) / 2)))
    print(get_time_str() + "Literal Number: " + str(int(len(kg_obj.get_lite_id_set()) / 2)))
    print(get_time_str() + "Relation Triple Number: " + str(int(len(kg_obj.get_relation_id_triples()) / 2)))
    print(get_time_str() + "Attribute Triple Number: " + str(int(len(kg_obj.get_attribute_id_triples()) / 2)))
    sys.stdout.flush()

def load_sbert(kg1_sbert_path, kg2_sbert_path):
    kg1_sbert_dict, kg2_sbert_dict = dict(), dict()
    with open(kg1_sbert_path, 'r') as f:
        reader = csv.reader(f)
        kg1_sbert_dict = {rows[0]: rows[1:] for rows in reader}
    with open(kg2_sbert_path, 'r') as f:
        reader = csv.reader(f)
        kg2_sbert_dict = {rows[0]: rows[1:] for rows in reader}
    return kg1_sbert_dict, kg2_sbert_dict

path = os.path.abspath(__file__)
base, _ = os.path.split(path)

# kg1_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_1")
# kg1_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_1")
kg1_rel_path = os.path.join(base, "data/KKdata_V4/vod_triplet.txt")
kg1_attr_path = os.path.join(base, "data/KKdata_V4/vod_triplet.txt")
kg1_sbert_path = os.path.join(base, "../pr/sbert_emb/vod_emb.csv")

# kg2_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_2")
# kg2_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_2")
kg2_rel_path = os.path.join(base, "data/KKdata_V4/tv_triplet.txt")
kg2_attr_path = os.path.join(base, "data/KKdata_V4/tv_triplet.txt")
kg2_sbert_path = os.path.join(base, "../pr/sbert_emb/tv_emb.csv")

# test_path = os.path.join(base, "data/MED-BBK-9K/ent_links")
test_path = os.path.join(base, "data/KKdata_V4/ent_mapping.txt")

print(get_time_str() + "Loading Sbert Embedding...")
sys.stdout.flush()

# load sbert embedding from pretrained model
kg1_sbert_dict, kg2_sbert_dict = load_sbert(kg1_sbert_path, kg2_sbert_path)

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

# construct KGs object
kgs = pu.construct_kgs(kg1, kg2)

# configure kgs
kgs.set_se_module(se.GCNAlign)
kgs.set_pr_module(pr.PARIS)

# Set Thread Number:
# kgs.pr.set_worker_num(6)

# Set Entity Candidate Number:
CANDIDATE_NUM = 1
kgs.pr.set_ent_candidate_num(CANDIDATE_NUM)
print(get_time_str() + "Entity Candidate Number: " + str(CANDIDATE_NUM))

# Set SBert Eqvalence Weight:
SBERT_EQV_WEIGHT = 0.9
kgs.pr.set_sbert_eqv_weight(SBERT_EQV_WEIGHT)
print(get_time_str() + "SBert Eqvalence Weight: " + str(SBERT_EQV_WEIGHT))

# Set Embedding Eqvalence Weight:
EMB_EQV_WEIGHT = 0.1
kgs.pr.set_emb_eqv_weight(EMB_EQV_WEIGHT)
print(get_time_str() + "Embedding Eqvalence Weight: " + str(EMB_EQV_WEIGHT))

ENT_EQV_THRE = 0
kgs.pr.set_ent_eqv_bar(ENT_EQV_THRE)
print(get_time_str() + "Entity Eqvalence Threshold: " + str(ENT_EQV_THRE))

PR_ITERATION_NUM = 10
kgs.pr.set_max_iteration_num(PR_ITERATION_NUM)
print(get_time_str() + "PR Iteration Number: " + str(PR_ITERATION_NUM))

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
kgs.test(test_path, threshold=[0.1 * i for i in range(10)])

kgs.pr.enable_rel_init(False)

iteration = 3
for i in range(iteration):
    print(get_time_str() + "Performing SE Module (GCNAlign)...")
    sys.stdout.flush()
    # run se module
    kgs.run_se(embedding_feedback=True, mapping_feedback=True)
    save_alignment(kgs, i+1, "SE")

    print_kgs_stat(kgs)
    kgs.test(test_path, threshold=[0.1 * i for i in range(10)])
    print(get_time_str() + "Performing PR Module (PARIS)...")
    sys.stdout.flush()
    # run pr module
    # if i == iteration-1:
    #     kgs.pr.set_ent_candidate_num(3)
    kgs.run_pr()
    save_alignment(kgs, i+1, "PR")

    print_kgs_stat(kgs)
    kgs.test(test_path, threshold=[0.1 * i for i in range(10)])

# Save alignment result:
# save_alignment(kgs)

# Save PRASE Model:
# save_path = os.path.join(base, "models/KKdata_V2/iter_1_model")
# save_path = os.path.join(base, "models/MED/test_model")
# pu.save_prase_model(kgs, save_path)
