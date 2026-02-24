import json
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(SCRIPT_DIR, '..', 'results', 'phase2_realtime_stats.json')

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("🚀 Monitor de Fase 2: Optimización Hiperparamétrica en Tiempo Real...")
    
    while True:
        if not os.path.exists(STATS_FILE):
            print("Esperando a que comience la optimización...", end='\r')
            time.sleep(1)
            continue
            
        try:
            with open(STATS_FILE, 'r') as f:
                data = json.load(f)
        except:
            time.sleep(0.5)
            continue

        clear_console()
        print("="*70)
        print(" 🧠 FASE 2: OPTIMIZACIÓN BA YESIANA (Optuna) - IEBS Alzheimer 🧠")
        print(f" Algoritmo Actual: {data.get('current_algo', 'N/A')}")
        print(f" Última actualización: {time.strftime('%H:%M:%S', time.localtime(data.get('last_update', 0)))}")
        print("="*70)
        
        results = data.get('results', {})
        if not results:
            print("Esperando resultados de las primeras pruebas (trials)...")
        else:
            print(f"{'Algoritmo':15} | {'Mejor Score':15} | {'Progreso':15} | {'Estado'}")
            print("-" * 70)
            for name, res in results.items():
                acc_str = f"{res['best_acc']:.4%}" if res['best_acc'] > 0 else "---"
                prog_str = f"Trial {res['trials']}/{res['total_trials']}"
                print(f"{name:15} | {acc_str:15} | {prog_str:15} | {res['status']}")
        
        print("\n" + "="*70)
        
        # Check if all completed
        all_done = all(r['status'] == "Completado" for r in results.values()) and len(results) == 3
        if all_done:
            print("\n✅ ¡Fase 2 de optimización terminada con éxito!")
            break
            
        time.sleep(2)

if __name__ == "__main__":
    main()
