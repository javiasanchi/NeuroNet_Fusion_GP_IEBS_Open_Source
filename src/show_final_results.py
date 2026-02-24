import json
import pandas as pd

def show_final_results():
    try:
        with open('Analytical_Biomarker_Project/results/phase2_realtime_stats.json', 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*45)
        print("🏆 RESULTADOS FINALES: OPTIMIZACIÓN FASE 2")
        print("="*45)
        print(f"{'Algoritmo':15} | {'Mejor Precisión':15}")
        print("-" * 45)
        for name, res in data['results'].items():
            print(f"{name:15} | {res['best_acc']:.4%}")
        print("="*45)
        
        # Determine the winner
        winner = max(data['results'].items(), key=lambda x: x[1]['best_acc'])
        print(f"\n🥇 EL CAMPEÓN ABSOLUTO ES: {winner[0]} ({winner[1]['best_acc']:.4%})")
        
    except Exception as e:
        print(f"Error reading stats: {e}")

if __name__ == "__main__":
    show_final_results()
