"""
Análisis consolidado de datos EDM
Incluye todos los análisis realizados en un solo script
"""

import pandas as pd
import os
import glob
from collections import defaultdict

def analyze_series_statistics():
    """Estadísticas de longitud de series por categoría."""
    print("\n" + "="*60)
    print("1. ESTADISTICAS DE LONGITUD DE SERIES")
    print("="*60)
    
    exclude_file = '8e1f46815d8d4169b6e8a83a1546e159'
    
    for category in ['NPT', 'Normal', 'OD']:
        print(f"\n{category}:")
        path = f"../Data/Option 2/Train/{category}"
        files = glob.glob(os.path.join(path, "*.csv"))
        
        lengths = []
        for file_path in files:
            filename = os.path.basename(file_path).replace('.csv', '')
            if filename != exclude_file:
                df = pd.read_csv(file_path)
                lengths.append(len(df))
        
        if lengths:
            print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")

def analyze_free_falling():
    """Análisis de Free Falling consecutivo."""
    print("\n" + "="*60)
    print("2. ANALISIS DE FREE FALLING")
    print("="*60)
    
    exclude_file = '8e1f46815d8d4169b6e8a83a1546e159'
    
    for category in ['NPT', 'Normal', 'OD']:
        print(f"\n{category}:")
        path = f"../Data/Option 2/Train/{category}"
        files = glob.glob(os.path.join(path, "*.csv"))
        
        lengths = []
        for file_path in files:
            filename = os.path.basename(file_path).replace('.csv', '')
            if filename != exclude_file:
                df = pd.read_csv(file_path)
                stages = df['Segment'].values
                
                max_length = 0
                current = 0
                for stage in stages:
                    if stage == 'Free Falling':
                        current += 1
                    else:
                        max_length = max(max_length, current)
                        current = 0
                max_length = max(max_length, current)
                
                if max_length > 0:
                    lengths.append(max_length)
        
        if lengths:
            print(f"  Con Free Falling: {len(lengths)}/{len(files)} archivos")
            print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")

def analyze_file_overlap():
    """Análisis de archivos compartidos entre opciones."""
    print("\n" + "="*60)
    print("3. ANALISIS DE ARCHIVOS COMPARTIDOS")
    print("="*60)
    
    paths = {
        'O1_Train': '../Data/Option 1/Train',
        'O1_Test': '../Data/Option 1/Test',
        'O2_Train': '../Data/Option 2/Train',
        'O2_Test': '../Data/Option 2/Test'
    }
    
    files_dict = {}
    for name, path in paths.items():
        files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        files_dict[name] = set(os.path.basename(f).replace('.csv', '') for f in files)
    
    print(f"\nOverlaps:")
    overlaps = [
        ('O1_Train', 'O2_Train', files_dict['O1_Train'] & files_dict['O2_Train']),
        ('O1_Test', 'O2_Test', files_dict['O1_Test'] & files_dict['O2_Test']),
    ]
    
    for name1, name2, intersection in overlaps:
        print(f"  {name1} & {name2}: {len(intersection)} archivos")

def analyze_free_falling_gaps():
    """Análisis de gaps entre segmentos Free Falling en OD."""
    print("\n" + "="*60)
    print("4. GAPS ENTRE SEGMENTOS FREE FALLING (OD)")
    print("="*60)
    
    path = '../Data/Option 2/Train/OD'
    files = glob.glob(os.path.join(path, "*.csv"))
    
    all_gaps = []
    for file_path in files:
        df = pd.read_csv(file_path)
        stages = df['Segment'].values
        
        segments = []
        start = None
        for i, stage in enumerate(stages):
            if stage == 'Free Falling':
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append({'start': start, 'end': i-1})
                    start = None
        if start is not None:
            segments.append({'start': start, 'end': len(stages)-1})
        
        # Calcular gaps
        for i in range(len(segments)-1):
            gap = segments[i+1]['start'] - segments[i]['end'] - 1
            all_gaps.append(gap)
    
    if all_gaps:
        print(f"  Total gaps: {len(all_gaps)}")
        print(f"  Min: {min(all_gaps)}, Max: {max(all_gaps)}, Avg: {sum(all_gaps)/len(all_gaps):.1f}")

def main():
    """Ejecutar todos los análisis."""
    print("ANALISIS CONSOLIDADO DE DATOS EDM")
    print("="*60)
    
    # Cambiar al directorio del script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    analyze_series_statistics()
    analyze_free_falling()
    analyze_file_overlap()
    analyze_free_falling_gaps()
    
    print("\n" + "="*60)
    print("ANALISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()

