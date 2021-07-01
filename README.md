- create folder to save the predictions
  - `mkdir predictions`

- baseline
  - `python baseline.py`
    
- anchor_only
  - only text
    - `python anchor_only.py -mode anchor_text_only -model_type pretrained`
  - only image
    - `python anchor_only.py -mode anchor_image_only -model_type pretrained`
  - text & image
    - `python anchor_only.py -mode anchor_text_image -model_type pretrained`

- complicated model
  - `python complicated_nn.py -mode all_bert_lstm -model_type pretrained`
  