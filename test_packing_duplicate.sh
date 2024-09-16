
model="mbv3_small"
# 169 211 254 296

depth_split_factor=5
layer_threshold=1


for num_crossbars in 249 299 349 399 449
do
ID="run-duplicate-crossbars:${num_crossbars}-${model}-heuristic-${layer_threshold}-split_factor:${depth_split_factor}"

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
done
