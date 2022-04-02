seed=0
enc_len=8
dec_len=24
lr=0.0001
wd=0.0001
gpu_id=(5 5 1 15 6 2)
# gpu_id=(7 13 9 10 11 12)
project_name='air_17'
group_name='rw_h16-64_g16_lr-4_wd-5_in1248'
model_array=('City_GC_GRU' 'GC_GRU' 'MTCity_GC_GRU' 'PM25GNN' 'HighAirv2' 'MT_GC_GRU' )
k_hop=1

date_path=../log/$(date +"%Y-%m-%d-%H-%M-%S")
mkdir -p $date_path

# rm log*
# nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[0]} --project-name $project_name --group-name $group_name --model ${model_array[0]} --k-hop 1 --run-idx 0 --seed $seed >> $date_path/log_0.txt &
# nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[1]} --project-name $project_name --group-name $group_name --model ${model_array[1]} --k-hop 1 --run-idx 1 --seed $seed >> $date_path/log_1.txt &
nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[2]} --project-name $project_name --group-name $group_name --model ${model_array[2]} --k-hop 1 --run-idx 2 --seed $seed >> $date_path/log_2.txt &
# nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[3]} --project-name $project_name --group-name $group_name --model ${model_array[3]} --k-hop 1 --run-idx 3 --seed $seed >> $date_path/log_3.txt &
# nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[4]} --project-name $project_name --group-name $group_name --model ${model_array[4]} --k-hop 1 --run-idx 4 --seed $seed >> $date_path/log_4.txt &
nohup python train.py --enc-len $enc_len --dec-len $dec_len --lr $lr --wd $wd --gpu-id ${gpu_id[5]} --project-name $project_name --group-name $group_name --model ${model_array[5]} --k-hop 1 --run-idx 5 --seed $seed >> $date_path/log_5.txt &
