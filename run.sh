for data in dbp_wd_15k_V1 dbp_yg_15k_V1
do
    for sbert in 0 0.1 0.2
    do
        for norm in -1.0 3.0 2.0 1.0
        do
        echo "$data, sbert $sbert, norm $norm:"
        python test/test.py --dataset $data --candidate_num 1 --sbert_eqv_weight $sbert --normalize_pow $norm
        done
    done
done