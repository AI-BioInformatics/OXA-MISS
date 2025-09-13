import torch
import os
import argparse
import importlib.util
import sys

def load_class_from_file(file_path, class_name):
    """Importa dinamicamente una classe da un file .py"""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def convert_checkpoints(old_class_file, old_class_name, new_class_file, new_class_name, input_dir, output_dir):
    # Carica le classi
    OldClass = load_class_from_file(old_class_file, old_class_name)
    NewClass = load_class_from_file(new_class_file, new_class_name)

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pt"):
            continue

        old_checkpoint_path = os.path.join(input_dir, filename)
        new_checkpoint_path = os.path.join(output_dir, filename)

        # Inizializza i modelli
        old_model = torch.load(old_checkpoint_path)

        # Estrai i pesi
        state_dict = old_model.state_dict()

        # Crea il nuovo modello e carica i pesi
        new_model = NewClass()
        new_model.load_state_dict(state_dict)

        torch.save(new_model, new_checkpoint_path)
        print(f"✅ Convertito: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converti checkpoint da una vecchia classe a una nuova")
    parser.add_argument('--old-class-file', type=str, default='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/experiments/models/Custom_Multimodal_XA_v2.py', help="Path al file .py contenente la vecchia classe")
    parser.add_argument('--old-class-name', type=str, default='Custom_Multimodal_XA_v2', help="Nome della vecchia classe")
    parser.add_argument('--new-class-file', type=str, default='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/experiments/models/OXA_MISS.py', help="Path al file .py contenente la nuova classe")
    parser.add_argument('--new-class-name', type=str, default='OXA_MISS', help="Nome della nuova classe")
    parser.add_argument('--input-dir', type=str, default='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo/chekpoints_KIRC_old', help="Directory con i vecchi checkpoint")
    parser.add_argument('--output-dir', type=str, default='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo/chekpoints_KIRC_new', help="Directory dove salvare i nuovi checkpoint")

    args = parser.parse_args()

    convert_checkpoints(
        old_class_file=args.old_class_file,
        old_class_name=args.old_class_name,
        new_class_file=args.new_class_file,
        new_class_name=args.new_class_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
