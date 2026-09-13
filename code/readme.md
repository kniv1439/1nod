An example of running script:

python main.py --backbone gcn --dataset ${DATASET} \
                --lr ${LR} --num_layers 2 \
                --hidden_channels ${dim} --weight_decay 5e-05 \
                --dropout ${dropout} --ours_weight_decay 0.\
                --method GPR --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
                --alpha 0.5  --device 3 --runs 5 --init_alpha 0.7 >> results/${DATASET}.txt

--backbone: Tn auxiliary GNN to capture the local information
--dataset: The employed dataset
--lr: learning rate
--num_layers: The number of layers for GNN
--hidden_channels: The hidden dimension in ParaFormer
--weight_decay: weight decay of GNN
--ours_weight_decay: weight decay of ParaFormer
--dropout: The dropout of entire model
--ours_use_residual: The use of residue connection
--alpha: The parameter utilized in residue connection
--init_alpha: utlized to initilize the paramter of PageRank