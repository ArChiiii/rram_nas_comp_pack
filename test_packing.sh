
model="mbv3_small"

depth_split_factor=1
layer_threshold=5
num_crossbars=20

ID="run-baseline-${model}2-input64-heuristic-${layer_threshold}-split_factor:${depth_split_factor}"
# ID="run-pack-${model}-crossbars:${num_crossbars}-split_factor:${depth_split_factor}-device-bit"
# ID="run-pack-${model}-split_factor:${depth_split_factor}-latency_sim-256"
# python test_parallel_packing.py --exp-id $ID \
python test_packing.py --exp-id $ID \
    --model $model \
    --depth-split-factor $depth_split_factor \
    --num-crossbars $num_crossbars\
    --layer-threshold $layer_threshold\
    --find-solution \
    --log-pack \
    --seed 1 \
    --verbose
