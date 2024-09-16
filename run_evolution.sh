source activate RRAM_PACK

depth_split_factor=1
layer_threshold=1
num_sample=8
num_crossbars=168 #211 #168 #253

ID="run-evolute-mbv3_small-cifar-population:50-num_crossbar:${num_crossbars}-heuristic-${layer_threshold}-depth_split_factor:${depth_split_factor}-num_sample:${num_sample}-NDGAII"

python main.py --exp-id $ID \
    --depth-split-factor $depth_split_factor \
    --num-sample $num_sample \
    --num-crossbars $num_crossbars \
    --layer-threshold $layer_threshold \
    --duplicate \
    --evolute-method "ndga" \
    --device 1 \
    --seed 1

