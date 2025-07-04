"""
Script per rinominare automaticamente i modelli nella cartella results/models/
in modo che abbiano il formato corretto: s_X.XX.zip o d_X.XX.zip
"""

import os
import re
import glob

def extract_noise_from_filename(filename):
    """
    Estrae il valore di noise dal nome del file.
    Supporta vari formati come: model_0.15.zip, s_0.15.zip, d_0.15.zip, etc.
    """
    # Rimuovi l'estensione
    name_without_ext = filename.replace('.zip', '')
    
    # Pattern per estrarre numeri decimali dal nome
    patterns = [
        r'(\d+\.\d+)',  # Formato X.XX
        r'(\d+)',       # Solo numero intero
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext)
        if match:
            return float(match.group(1))
    
    return 0.0  # Default se non trova niente

def determine_model_type(filename):
    """
    Determina se il modello è statico (s_) o dinamico (d_) basandosi sul nome.
    """
    filename_lower = filename.lower()
    
    # Se già ha il prefisso corretto, mantienilo
    if filename_lower.startswith('s_'):
        return 's'
    elif filename_lower.startswith('d_'):
        return 'd'
    
    # Altrimenti cerca indizi nel nome
    dynamic_indicators = ['dyn', 'dynamic', 'dinamic', 'd_', 'adapt']
    static_indicators = ['stat', 'static', 's_', 'normal']
    
    for indicator in dynamic_indicators:
        if indicator in filename_lower:
            return 'd'
    
    for indicator in static_indicators:
        if indicator in filename_lower:
            return 's'
    
    # Default: considera statico se noise = 0.0, altrimenti dinamico
    noise = extract_noise_from_filename(filename)
    return 's' if noise == 0.0 else 'd'

def rename_models_in_directory(directory_path, dry_run=True):
    """
    Rinomina tutti i modelli in una directory specifica.
    
    Args:
        directory_path (str): Percorso della directory
        dry_run (bool): Se True, mostra solo cosa farebbe senza effettuare i cambiamenti
    """
    if not os.path.exists(directory_path):
        print(f"❌ Directory non trovata: {directory_path}")
        return
    
    # Trova tutti i file nella directory (non solo .zip)
    all_files = [f for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f)) 
                 and not f.startswith('.')]  # Escludi file nascosti
    
    if not all_files:
        print(f"📁 Nessun file trovato in: {directory_path}")
        return
    
    print(f"📁 Elaborando directory: {directory_path}")
    print(f"📄 Trovati {len(all_files)} file")
    
    renamed_count = 0
    
    for filename in all_files:
        file_path = os.path.join(directory_path, filename)
        
        # Salta se è già un file .zip
        if filename.endswith('.zip'):
            print(f"✅ {filename} → già ha estensione .zip")
            continue
        
        # Estrai informazioni dal nome corrente
        noise_value = extract_noise_from_filename(filename)
        model_type = determine_model_type(filename)
        
        # Genera il nuovo nome CON l'estensione .zip
        new_filename = f"{model_type}_{noise_value:.2f}.zip"
        new_path = os.path.join(directory_path, new_filename)
        
        # Controlla se il file di destinazione esiste già
        if os.path.exists(new_path):
            print(f"⚠️  {filename} → {new_filename} (SALTATO: file destinazione già esistente)")
            continue
        
        print(f"🔄 {filename} → {new_filename}")
        
        if not dry_run:
            try:
                os.rename(file_path, new_path)
                renamed_count += 1
                print(f"   ✅ Rinominato con successo")
            except Exception as e:
                print(f"   ❌ Errore durante la rinominazione: {e}")
        else:
            renamed_count += 1
    
    if dry_run:
        print(f"\n🔍 DRY RUN: {renamed_count} file sarebbero stati rinominati")
        print("💡 Esegui con dry_run=False per effettuare i cambiamenti")
    else:
        print(f"\n✅ Rinominazione completata: {renamed_count} file modificati")

def main():
    """Funzione principale per rinominare tutti i modelli."""
    
    # Percorso base della cartella models
    base_path = os.path.join("results", "models")
    
    if not os.path.exists(base_path):
        print(f"❌ Cartella base non trovata: {base_path}")
        print("💡 Assicurati di eseguire lo script dalla directory root del progetto")
        return
    
    print("🔧 SCRIPT RINOMINAZIONE MODELLI")
    print("=" * 50)
    
    # Chiedi conferma all'utente
    print(f"📁 Cartella base: {os.path.abspath(base_path)}")
    
    # Prima esegui un dry run per mostrare cosa cambierebbe
    print("\n🔍 ANTEPRIMA MODIFICHE (Dry Run):")
    print("-" * 30)
    
    # Trova tutte le sottocartelle (seeds)
    seed_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    
    if not seed_dirs:
        print(f"❌ Nessuna cartella seed trovata in {base_path}")
        return
    
    print(f"📂 Trovate {len(seed_dirs)} cartelle seed: {seed_dirs}")
    
    # Dry run su tutte le cartelle
    for seed_dir in seed_dirs:
        seed_path = os.path.join(base_path, seed_dir)
        rename_models_in_directory(seed_path, dry_run=True)
        print()
    
    # Chiedi conferma per procedere
    response = input("\n❓ Vuoi procedere con la rinominazione? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', 'si', 's']:
        print("\n🚀 Esecuzione rinominazione...")
        print("-" * 30)
        
        for seed_dir in seed_dirs:
            seed_path = os.path.join(base_path, seed_dir)
            rename_models_in_directory(seed_path, dry_run=False)
            print()
        
        print("🎉 Rinominazione completata!")
    else:
        print("❌ Operazione annullata dall'utente")

if __name__ == "__main__":
    main()