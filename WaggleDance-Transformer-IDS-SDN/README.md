# WaggleDance-Transformer-IDS

This repository contains the implementation of the **WaggleMoE-TabTransformer** intrusion detection model
(Waggle Dance–inspired Mixture-of-Experts Transformer) on the InSDN dataset.

The file `src/wagglemoe_insdn.py` contains the full training pipeline:
- loading and cleaning InSDN CSV files
- preprocessing + mutual information feature selection
- definition of the WaggleMoE-TabTransformer model (novelty)
- training loop with epsilon-annealing and load-balance loss
- threshold tuning and final evaluation

## How to use

1. Place your InSDN CSV files under:

   ```
   data/InSDN_DatasetCSV/
       Normal_data.csv
       metasploitable-2.csv
       OVS.csv
   ```

2. Install dependencies:

   ```bash
   pip install -r environment/requirements.txt
   ```

3. Run the script:

   ```bash
   python src/wagglemoe_insdn.py
   ```

Adjust paths or parameters inside the script if needed.
