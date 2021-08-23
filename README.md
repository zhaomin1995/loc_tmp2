- create folder to save the predictions
  - `mkdir predictions`

- Majority baseline
  - `python baseline.py`
    
- Neural network only using anchor tweet (neural network baselines)
  - only use text
    - `python anchor_only.py -mode anchor_text_only -model_type pretrained`
  - only use image
    - `python anchor_only.py -mode anchor_image_only -model_type pretrained`
  - use text & image
    - `python anchor_only.py -mode anchor_text_image -model_type pretrained`

- Context-aware neural network
  - use tweets_before & tweets_after
    - `python complicated_nn.py -mode all_bert_lstm -model_type pretrained`
  - use only tweets_before
    - `python complicated_nn.py -mode all_bert_lstm_onlybefore -model_type pretrained`
  - use only tweets_after
    - `python complicated_nn.py -mode all_bert_lstm_onlyafter -model_type pretrained`
  - ignore additional textual features
    - `python complicated_nn.py -mode all_bert_lstm_noaddfeat -model_type retrain`