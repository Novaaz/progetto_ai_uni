import generate_models
import analyze_models
import ensemble_training
import testing_on_new_env
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.constants import *
from src.utils.core import clean_dead_dir

def show_simple_menu():
	"""
	Menu semplice
	"""
	print("\n" + "="*50)
	print("🎯 CITYLEARN AI PROJECT")
	print("="*50)
	print("1. 🚀 Pipeline completa")
	print("2. 🤖 Solo Generate Models")
	print("3. 📈 Solo Analyze Models")
	print("4. 🔧 Solo Ensemble Training")
	print("5. 🧪 Test Random Environment")
	print("6. 🧹 Pulisci directory vuote")
	print("0. ❌ Esci")
	print("="*50)

def main():
	"""
	Menu principale semplice
	"""
	while True:
		show_simple_menu()
		
		try:
			choice = input("\n🔧 Seleziona: ").strip()
			
			if choice == '0':
				print("\n👋 Ciao!")
				break
				
			elif choice == '1':
				print("\n🚀 === PIPELINE COMPLETA ===")
				generate_models.main()
				stats_path, best_models_path = analyze_models.main()
				if stats_path and best_models_path and os.path.exists(best_models_path):
					print(f"\n✅ File analisi creato: {best_models_path}")
					ensemble = ensemble_training.main()
					if ensemble:
						print("\n🎯 Pipeline completa eseguita con successo!")
					else:
						print("\n❌ Errore durante ensemble training")
				else:
					print("\n❌ Errore durante l'analisi")
					
			elif choice == '2':
				print("\n🤖 === GENERATE MODELS ===")
				generate_models.main()
				print("\n✅ Generazione completata!")
				
			elif choice == '3':
				print("\n📈 === ANALYZE MODELS ===")
				stats_path, best_models_path = analyze_models.main()
				if stats_path and best_models_path:
					print(f"\n✅ Analisi completata: {best_models_path}")
				else:
					print("\n❌ Errore durante analisi")
					
			elif choice == '4':
				print("\n🔧 === ENSEMBLE TRAINING ===")
				ensemble = ensemble_training.main()
				if ensemble:
					print("\n✅ Ensemble training completato!")
				else:
					print("\n❌ Errore durante ensemble training")
					
			elif choice == '5':
				print("\n🧪 === TEST RANDOM ENVIRONMENT ===")
				testing_on_new_env.main()
				
			elif choice == '6':
				print("\n🧹 Pulendo directory vuote...")
				clean_dead_dir()
				print("✅ Pulizia completata!")
				
			else:
				print("❌ Opzione non valida!")
				
		except KeyboardInterrupt:
			print("\n\n👋 Uscita...")
			break
		except Exception as e:
			print(f"\n❌ Errore: {e}")
		
		input("\n⏸️  Premi Enter per continuare...")

if __name__ == "__main__":
	main()
