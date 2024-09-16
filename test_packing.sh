
model="mbv3_small"

depth_split_factor=5
layer_threshold=1
num_crossbars=20

ID="run-baseline-${model}-heuristic-${layer_threshold}-split_factor:${depth_split_factor}"

python test_packing.py --exp-id $ID \
    --model $model \
    --depth-split-factor $depth_split_factor \
    --num-crossbars $num_crossbars\
    --layer-threshold $layer_threshold\
    --find-solution \
    --log-pack \
    --seed 123 \
    --verbose
