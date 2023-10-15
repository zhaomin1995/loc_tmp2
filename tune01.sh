# flan_ul2 + target
CUDA_VISIBLE_DEVICES=0,1 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
CUDA_VISIBLE_DEVICES=0,1 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot
CUDA_VISIBLE_DEVICES=0,1 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot
CUDA_VISIBLE_DEVICES=0,1 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot
