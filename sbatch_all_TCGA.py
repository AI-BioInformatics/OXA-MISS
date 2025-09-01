import subprocess
import time

# Define parameters
TCGAs = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'KIRC', 'KIRP', 'LUAD', 'LUSC', 'OV', 'STAD']
# TCGAs = ['BLCA']

# TCGAs = ['BLCA', 'KIRC', 'LUAD']
seeds = [43]

# Template for sed command to modify the SLURM script
sed_template = "sed 's/--seed [0-9]\\+/--seed {}/' slurm_templates/MUSE.sh | sed 's/--TCGA_dataset_name [A-Z]\\+/--TCGA_dataset_name {}/' > temp_script.sh"

# Loop through all combinations
for TCGA_id in TCGAs:
    for seed in seeds:
        # Create temporary script with new parameters
        sed_command = sed_template.format(seed, TCGA_id)
        subprocess.run(sed_command, shell=True)
        
        # Submit the job
        subprocess.run(['sbatch', 'temp_script.sh'])
        
        # Wait a bit to avoid overwhelming the scheduler
        time.sleep(0.5)
        
        # Clean up
        subprocess.run(['rm', 'temp_script.sh'])

print("All jobs submitted successfully!")