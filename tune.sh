# flan-t5 + zero-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot

# flan-t5 + few-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot

# flan-ul2 + zero-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot

# flan-ul2 + few-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_ul2 -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot

# flan-alpaca + zero-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar zero-shot

# flan-alpaca + few-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content early_target -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content target_later -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot
python llm.py -data_dir data/ -experiment flan_alpaca -input_content all -output_dir output -cache_dir /mnt/DATA/hf_cache/ -exemplar few-shot

