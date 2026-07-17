# -*- coding: utf-8 -*-
"""Train all 5 folds for human Brain model"""
import os, subprocess, shutil, glob, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

results = {}
for fold in range(1, 6):
    print(f'\n=== Training Fold {fold} ===')
    cmd = f'python train.py --species human --tissue Brain --fold {fold} --epochs 100 --patience 15 --device cpu'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Find the latest result directory
    out_dirs = sorted(glob.glob('../train_results/human_Brain_fold_*'))
    latest = out_dirs[-1] if out_dirs else None
    
    if latest and os.path.exists(os.path.join(latest, 'metrics.json')):
        m = json.load(open(os.path.join(latest, 'metrics.json')))
        results[f'fold_{fold}'] = m
        print(f'  AUC={m["auc"]:.4f}, AUPR={m["aupr"]:.4f}')
        
        # Copy best model to train dir
        if os.path.exists(os.path.join(latest, 'best_model.pth')):
            shutil.copy2(os.path.join(latest, 'best_model.pth'), f'best_model_fold{fold}.pth')
    else:
        print(f'  No metrics found')
        results[f'fold_{fold}'] = {'error': 'no metrics'}

# Save summary
json.dump(results, open('human_Brain_5fold_results.json', 'w'), indent=2)
print('\n=== 5-Fold Complete ===')
for f, m in results.items():
    if 'auc' in m:
        print(f'{f}: AUC={m["auc"]:.4f}, AUPR={m["aupr"]:.4f}')
    else:
        print(f'{f}: {m}')
