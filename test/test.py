import os
import sys
import pr
import se
import sb
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


def save_alignment(kgs_obj, iter, module):
    # output_dir = f"output/KKdata_V4/last_multi/ent_cand_3"
    # output_dir = f"output/KKdata_V4/test_new_title_filter"
    output_dir = f"output/KKdata_V4/sbert_weight/sbert_weight_{SBERT_EQV_WEIGHT}/set_ent_candidate_{CANDIDATE_NUM}"
    # output_dir = "output/TopK_test/test_30_thre_0.1"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{module}_iter_{iter}.txt")
    with open(output_path, 'w') as fout:
        for tuple in kgs_obj.get_ent_align_name_result():
            if tuple[0] != "" and tuple[1] != "" and tuple[0] in kg1_head_set:
                fout.write(tuple[0]+'\t'+tuple[1]+'\t'+str(tuple[2])+'\n')

def load_triple_head(rel_path_1, rel_path_2):
    kg1_head_set, kg2_head_set = set(), set()
    with open(rel_path_1, "r") as f:
        data = f.readlines()
        for row in data:
            kg1_head_set.add(row.strip().split('\t')[0])
    with open(rel_path_2, "r") as f:
        data = f.readlines()
        for row in data:
            kg2_head_set.add(row.strip().split('\t')[0])
    return kg1_head_set, kg2_head_set

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

# kg1_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_1")
# kg1_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_1")
kg1_rel_path = os.path.join(base, "data/KKdata_V4/vod_triplet.txt")
kg1_attr_path = os.path.join(base, "data/KKdata_V4/vod_triplet.txt")

# kg2_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_2")
# kg2_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_2")
kg2_rel_path = os.path.join(base, "data/KKdata_V4/tv_triplet.txt")
kg2_attr_path = os.path.join(base, "data/KKdata_V4/tv_triplet.txt")

# test_path = os.path.join(base, "data/MED-BBK-9K/ent_links")
test_path = os.path.join(base, "data/KKdata_V4/ent_mapping.txt")

kg1_head_set, kg2_head_set = load_triple_head(kg1_rel_path, kg2_rel_path)

print(get_time_str() + "Japanese Sentence Bert Inferencing...")
sys.stdout.flush()

SBert = sb.SBert()
kg1_sbert_dict, kg2_sbert_dict = SBert.get_sbert_dict(kg1_head_set, kg2_head_set)

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
CANDIDATE_NUM = 1
kgs.pr.set_ent_candidate_num(CANDIDATE_NUM)
print(get_time_str() + "Entity Candidate Number: " + str(CANDIDATE_NUM))

# Set SBert Eqvalence Weight:
SBERT_EQV_WEIGHT = 0.1
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

NORMALIZE_POW = 1
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
kgs.test(test_path, threshold=[0.1 * i for i in range(10)])

kgs.pr.enable_rel_init(False)

iteration = 1
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

# Save PRASE Model:
# save_path = os.path.join(base, "models/KKdata_V4/PRASE_model")
# pu.save_prase_model(kgs, save_path)
