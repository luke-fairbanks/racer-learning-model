# Train a fresh base model with the new physics (run in background)

nohup python3 run.py train_base --steps 3000000 &

# Then fine-tune to monaco

python3 run.py fine_tune --base models/\_base/<new_model>.zip --track monaco --steps 500000
