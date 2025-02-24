
# kuralGPT
 This is a fun project based on nanoGPT, that generates Thirukkural in transliteration format. I am using a simple data set from hugging face. 
 
Steps to install and run this repo.

##### Clone the Repo:
    https://github.com/jb-balaji/kuralGPT
    
##### Create python virtual env
	python3 -m venv kuralgpt_env

##### Activate venv
	source kuralgpt_env/bin/activate
	
##### Install dependendices 
	pip install torch numpy transformers datasets tiktoken wandb tqdm polars
	

##### Prepare the data
	python data/tirukkural_hf/prepare_hf.py 
	
##### Train the dataset
	python train.py config/train_tirukkural_hf.py --device=cpu --compile=False --eval_iters=20 --log_interval=10 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
	

##### Run sample 
Change the "keyword" to any tamil transliteration words like Aram, Mazhai, Inbam, Kaadhal etc.,

	    python sample.py --out_dir=out-tirukkural-hf --device=cpu --start="Anbu"
 
     

##### Acknowledgements

- nanoGPT : https://github.com/karpathy/nanoGPT
- Dataset : Selvakumarduraipandian/Thirukural
