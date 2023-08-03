b7:
	accelerate launch train.py --max_steps 200 --model_name ~/llama/converted-llama-2-7b-chat/

b13:
	accelerate launch train.py --max_steps 1000 --model_name ~/llama/converted-llama-2-13b-chat/

b70:
	accelerate launch train.py --max_steps 200 --model_name ~/llama/converted-llama-2-70b-chat/


inference:
	python inference.py --dir results/005_finetune
