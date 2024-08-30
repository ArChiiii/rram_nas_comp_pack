
model="mbv3_small"
# 169 211 254 296

depth_split_factor=5
layer_threshold=5

# for seed in 123 456 789
# do
# for num_crossbars in 698 838 978 1117 1120 1257
# for num_crossbars in 700 1050 1400 1750 2100
for num_crossbars in 168 211 254 296 338
# for num_crossbars in 170 255 340 425 510
# for num_crossbars in 80 96 112 128 144
# for num_crossbars in 80 120 160 200 240
do
ID="run-duplicate-crossbars:${num_crossbars}-${model}-heuristic-${layer_threshold}-split_factor:${depth_split_factor}-multiple-pack-no-max-load"
# "
# ID="run-pack-${model}-crossbars:${num_crossbars}-split_factor:${depth_split_factor}-device-bit"
# ID="run-pack-${model}-split_factor:${depth_split_factor}-latency_sim-256"
# python test_parallel_packing.py --exp-id $ID \
python test_packing_duplicate.py --exp-id $ID \
    --model $model \
    --depth-split-factor $depth_split_factor \
    --num-crossbars $num_crossbars\
    --layer-threshold $layer_threshold\
    --duplicate \
    --log-pack \
    --seed 123 \
    --device 0 \
    --verbose
    # --duplicate \
done
# done
