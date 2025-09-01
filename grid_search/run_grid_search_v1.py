import subprocess, json, os, time

def run_grid_search(sh_path, grid_search_versions_path):
    # Leggi il file JSON
    with open(grid_search_versions_path, 'r') as f:
        grid_search_versions = json.load(f)


    # Itera su ogni indice della lista
    for i in range(len(grid_search_versions)):
    # for i in range(1, len(grid_search_versions)):

    # for i in [0]:
        time.sleep(0.5)
        subprocess.run(["sbatch", sh_path, "--grid_search_model_version_index", str(i)], check=True)

if __name__ == "__main__":
    model_name = "MUSE" #Custom_Multimodal_XA or MUSE
    sh_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/grid_search/run_grid_search_v1.sh"
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
    run_grid_search(sh_path, grid_search_versions_path)
