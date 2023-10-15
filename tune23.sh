# flan_ul2 + target_later
CUDA_VISIBLE_DEVICES=2,3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
CUDA_VISIBLE_DEVICES=2,3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar one-shot
CUDA_VISIBLE_DEVICES=2,3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar five-shot
CUDA_VISIBLE_DEVICES=2,3 python llm.py -data_dir data/ -experiment flan_ul2+ -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar ten-shot
