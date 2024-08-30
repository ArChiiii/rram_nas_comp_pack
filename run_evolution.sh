source activate RRAM_PACK

depth_split_factor=1
num_sample=8
num_crossbars=253 #211 #168 #253
layer_threshold=1
# margin_factor=0.9
# coef_acc=5
# coef_latency=30
# -LP_duplicate
ID="run-evolute-mbv3_small-cifar-population:50-num_crossbar:${num_crossbars}-heuristic-${layer_threshold}-depth_split_factor:${depth_split_factor}-num_sample:${num_sample}-NDGAII-child-duplicate2-latency2"
# ID="run-evolute-mbv3_small-population:50-num_crossbar:${num_crossbars}-depth_split_factor:${depth_split_factor}-num_sample:${num_sample}_fitness${coef_acc}:${coef_latency}-LP_duplicate-FT:1"
python main.py --exp-id $ID \
    --depth-split-factor $depth_split_factor \
    --num-sample $num_sample \
    --num-crossbars $num_crossbars \
    --layer-threshold $layer_threshold \
    --duplicate \
    --evolute-method "ndga" \
    --device 1 \
    --seed 1
    # --resume \
    # --no-fine-tune \
    # --coef-acc $coef_acc \
    # --coef-latency $coef_latency \

    # --margin-factor $margin_factor \
# depth_split_factor=10
# num_sample=1
# num_crossbars=300

# ID="run-evolute-mbv3_small-num_crossbar:${num_crossbars}-depth_split_factor:${depth_split_factor}-num_sample:${num_sample}_fitness5:1"
# # ID="debug-evolute-consuming-process-num_crossbar:${num_crossbars}-depth_split_factor:${depth_split_factor}-num_sample:${num_sample}"

# python main.py --exp-id $ID \
#     --depth-split-factor $depth_split_factor \
#     --num-sample $num_sample \
#     --num-crossbars $num_crossbars &
