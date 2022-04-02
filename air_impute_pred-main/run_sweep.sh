date_path=../sweep_log/$(date +"%Y-%m-%d-%H-%M-%S")
mkdir -p $date_path

wandb_name=tokaka/air_sweep_test/1hz6et4n

# CUDA_VISIBLE_DEVICES=0 wandb agent $wandb_name > $date_path/sweep0.txt &
# CUDA_VISIBLE_DEVICES=1 wandb agent $wandb_name > $date_path/sweep1.txt &
# CUDA_VISIBLE_DEVICES=2 wandb agent $wandb_name > $date_path/sweep2.txt &
# CUDA_VISIBLE_DEVICES=3 wandb agent $wandb_name > $date_path/sweep3.txt &
# CUDA_VISIBLE_DEVICES=4 wandb agent $wandb_name > $date_path/sweep4.txt &
# CUDA_VISIBLE_DEVICES=5 wandb agent $wandb_name > $date_path/sweep5.txt &
# CUDA_VISIBLE_DEVICES=6 wandb agent $wandb_name > $date_path/sweep6.txt &
# CUDA_VISIBLE_DEVICES=7 wandb agent $wandb_name > $date_path/sweep7.txt &
CUDA_VISIBLE_DEVICES=8 wandb agent $wandb_name > $date_path/sweep8.txt &
# CUDA_VISIBLE_DEVICES=9 wandb agent $wandb_name > $date_path/sweep9.txt &
# CUDA_VISIBLE_DEVICES=10 wandb agent $wandb_name > $date_path/sweep10.txt &
# CUDA_VISIBLE_DEVICES=11 wandb agent $wandb_name > $date_path/sweep11.txt &
# CUDA_VISIBLE_DEVICES=12 wandb agent $wandb_name > $date_path/sweep12.txt &
CUDA_VISIBLE_DEVICES=13 wandb agent $wandb_name > $date_path/sweep13.txt &
# CUDA_VISIBLE_DEVICES=14 wandb agent $wandb_name > $date_path/sweep14.txt &
# CUDA_VISIBLE_DEVICES=15 wandb agent $wandb_name > $date_path/sweep15.txt &

# nohup some_command > /dev/null 2>&1&