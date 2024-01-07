b7:
	accelerate launch supervised_pretrain.py --max_steps 2000 --model_name /mnt/data/llama/converted-7b/ --use_8bit

p7:
	python supervised_pretrain.py --max_steps 2000 --model_name /mnt/data/llama/converted-7b/ --use_8bit --save_steps 1


second_layer:
	python rl_finetune.py --load_dir=results/123_supervised_pretrain/checkpoint-500/ --easy  \
	--model_name=/mnt/data/llama/converted-7b/ --max_steps 20000  \
	--depth_curriculum --curriculum_index 1 \
	--trainer pg  \
	--bf16  --use_8bit \
	--per_device_train_batch_size 1 \
	--shrink_head

	
b7rl:
	python rl_finetune.py --load_dir=results/123_supervised_pretrain/checkpoint-500/ --easy  \
	--model_name=/mnt/data/llama/converted-7b/ --max_steps 20000  \
	--depth_curriculum  \
	--trainer pg  \
	--bf16  --use_8bit \
	--per_device_train_batch_size 1 \
	--shrink_head
	

r7:
	python rl_finetune.py --use_8bit --max_new_tokens=2 --bf16 --model_name=/mnt/data/llama/converted-7b/ --learning_rate 0.0001
	# python rl_finetune.py --use_8bit --max_new_tokens=2 --bf16 --model_name=/mnt/data/llama/converted-7b/ --learning_rate 0.0001 --load_dir=./results/098_supervised_pretrain/checkpoint-200/

gen_data:
	python data_gen/gen_auto_explore_markov.py --model_name /mnt/data/llama/converted-7b/

b13:
	accelerate launch train.py --max_steps 1000 --model_name ~/llama/converted-llama-2-13b-chat/

b70:
	accelerate launch train.py --max_steps 200 --model_name ~/llama/converted-llama-2-70b-chat/

inference:
	python inference.py --dir results/005_finetune

host:
	python host.py --dir results/003_finetune

manual_7b:
	python inference.py --dir results/003_finetune --mode manual

manual_70b:
	python inference.py --dir results/005_finetune --mode manual

format:
	# yapf --in-place --style=google *.py
	# yapf --in-place --style=google **/**.py
	# yapf -ir --style=google .
	find . -name '*.py' -print0 | xargs -0 yapf -i
