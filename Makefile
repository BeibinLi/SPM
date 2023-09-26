
default:
	python train.py --max_steps 200 --model_name microsoft/phi-1_5 --use_8bit


b7:
	accelerate launch train.py --max_steps 200 --model_name /mnt/data/llama/converted-7b-chat/ --use_8bit


phi:
	accelerate launch train.py --max_steps 200 --model_name microsoft/phi-1_5 --use_8bit

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
