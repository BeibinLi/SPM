b7:
	accelerate launch train.py --max_steps 200 --model_name ~/llama/converted-llama-2-7b-chat/

b70:
	accelerate launch train.py --max_steps 200 --model_name ~/llama/converted-llama-2-70b-chat/


inference:
    python inference.py --dir results/001_finetune
