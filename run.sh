condor_rm hhao9

python3 gen_combo.py

rm -rf error_logs/*
rm -rf logs/*

for dir in pysubebm pysubebm_with_labels sustain_gmm sustain_kde; do 
    # Delete all files in subdirectories (depth ≥ 2)
    find "algo_results/$dir" -mindepth 2 -type f -delete
done 

# for dir in pysubebm pysubebm_with_labels; do 
#     # Delete all files in subdirectories (depth ≥ 2)
#     find "algo_results/$dir" -mindepth 2 -type f -delete
# done 

# for dir in sustain_gmm sustain_kde; do 
#     # Delete all files in subdirectories (depth ≥ 2)
#     find "algo_results/$dir" -mindepth 2 -type f -delete
# done 

condor_submit /home/hhao9/subtypes/run_mlhc.sub