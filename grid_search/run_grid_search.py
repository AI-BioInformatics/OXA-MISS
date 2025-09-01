import subprocess, json, os, time
import argparse




def run_grid_search(sh_path, grid_search_versions_path, model_name):
    # Leggi il file JSON
    with open(grid_search_versions_path, 'r') as f:
        grid_search_versions = json.load(f)


    # Itera su ogni indice della lista
    for i in range(0, len(grid_search_versions)):
    # for i in [8,9]:
        time.sleep(0.5)
        tumor= grid_search_versions[i]['data_loader']['KFold']['splits'].split('/')[-1]
        subprocess.run(["sbatch","--output", f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/{model_name}_augtrain_%j_{tumor}.out", sh_path, "--grid_search_model_version_index", str(i), "--model_name", model_name], check=True)
 
if __name__ == "__main__":
    #Prendi model name da riga di comando
    parser = argparse.ArgumentParser(description='Run grid search for model versions.')
    parser.add_argument('--model_name', type=str, required=True, default='MUSE', help='Name of the model (e.g., Custom_Multimodal_XA or MUSE).')
    model_name = parser.parse_args().model_name
    sh_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/grid_search/run_grid_search.sh"
    grid_search_versions_path = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/grid_search/models_versions/{model_name}_versions.json"

    if not os.path.isfile(sh_path):
        raise FileNotFoundError(f"Il file {sh_path} non esiste.")
    if not sh_path.endswith(".sh"):
        raise ValueError(f"Il file {sh_path} non è un file .sh.")
    
    if not os.path.isfile(grid_search_versions_path):
        raise FileNotFoundError(f"Il file {grid_search_versions_path} non esiste.")
    if not grid_search_versions_path.endswith(".json"):
        raise ValueError(f"Il file {grid_search_versions_path} non è un file JSON.")
    # Esegui la funzione con i parametri forniti
    run_grid_search(sh_path, grid_search_versions_path, model_name)
