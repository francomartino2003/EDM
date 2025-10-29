import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil

def plot_voltage_z_by_stage(df, xcol=None, vcol='Voltage', zcol='Z', scol='stage', stage_colors=None, title='EDM drill run'):
    """Grafica voltaje (izq), Z (der) y fondos por stage para un df con columnas [vcol, zcol, scol]."""
    # --- Eje X ---
    if xcol is None:
        x = range(len(df))
        xlabel = 'sample'
    else:
        x = df[xcol].to_numpy()
        xlabel = xcol

    # --- Señales ---
    v = df[vcol].to_numpy()
    z = df[zcol].to_numpy()

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # Solo agregar fondos por stage si existe la columna stage y tiene datos
    if scol in df.columns and df[scol].notna().any():
        # Stages (relleno por tramos contiguos)
        stages = pd.Series(df[scol]).ffill().astype(str)
        change_idx = [0] + list(stages[stages != stages.shift()].index) + [len(df)]
        segments = [(change_idx[i], change_idx[i+1], stages.iloc[change_idx[i]]) for i in range(len(change_idx)-1)]

        # Colores por defecto si no se especifican
        if stage_colors is None:
            unique_stages = list(set([s for _,_,s in segments]))
            base = ['#e8eaf6', '#e6f4ea', '#fff4e5', '#fde8e8', '#f3e8ff', '#e0f2fe']
            stage_colors = {st: base[i % len(base)] for i, st in enumerate(unique_stages)}

        # Fondos por stage
        for i0, i1, st in segments:
            if i1 <= len(x):
                ax1.axvspan(x[i0], x[i1-1], facecolor=stage_colors.get(st, '#eeeeee'), alpha=0.6, zorder=0)
                # etiqueta centrada arriba del tramo
                xm = 0.5 * (x[i0] + x[i1-1])
                ax1.text(xm, 0.98, st, transform=ax1.get_xaxis_transform(), ha='center', va='top', fontsize=9, alpha=0.9)

    # Curvas
    l1, = ax1.plot(x, v, linewidth=1.2, label=vcol, color='#1f77b4')
    l2, = ax2.plot(x, z, linewidth=1.4, linestyle='-', label=zcol, color='#ff7f0e')

    # Estética
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Voltage', color='#1f77b4')
    ax2.set_ylabel('Z', color='#ff7f0e')
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)

    # Leyenda combinada
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    return fig, (ax1, ax2)

def create_data_viz_structure():
    """
    Crear estructura Data Viz con visualizaciones de Option 1 y Option 2
    """
    
    print("CREANDO ESTRUCTURA DATA VIZ")
    print("=" * 50)
    
    # Crear directorio principal
    viz_base = "Data Viz"
    if os.path.exists(viz_base):
        shutil.rmtree(viz_base)
    os.makedirs(viz_base)
    
    # Configuración de colores para Option 2 (con stages)
    stage_colors = {
        'Touching': '#dbeafe',
        'Pre-drilling': '#e6f4ea',
        'Body Drilling': '#fff4e5',
        'Break-through': '#fde8e8',
        'Free Falling': '#ff4444',
        'Rework Free Falling': '#8B0000',
        'Retraction': '#f3e8ff',
        'Scarfing': '#e0f2fe',
        'Drilling': '#f0f0f0',
        'Machine Health Issue': '#ffcc00'
    }
    
    options = ['Option 1', 'Option 2']
    splits = ['Train', 'Test']
    
    total_plots = 0
    
    for option in options:
        print(f"\nPROCESANDO {option}:")
        print("-" * 30)
        
        option_path = f"Data/{option}"
        viz_option_path = f"{viz_base}/{option}"
        
        if not os.path.exists(option_path):
            print(f"  ERROR: {option_path} no existe")
            continue
            
        os.makedirs(viz_option_path, exist_ok=True)
        
        for split in splits:
            print(f"\n  {split}:")
            
            split_path = f"{option_path}/{split}"
            viz_split_path = f"{viz_option_path}/{split}"
            
            if not os.path.exists(split_path):
                print(f"    ERROR: {split_path} no existe")
                continue
                
            os.makedirs(viz_split_path, exist_ok=True)
            
            # Buscar todas las subcarpetas (categorías)
            categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            
            for category in categories:
                print(f"    Categoria: {category}")
                
                category_path = f"{split_path}/{category}"
                viz_category_path = f"{viz_split_path}/{category}"
                os.makedirs(viz_category_path, exist_ok=True)
                
                # Buscar archivos CSV
                csv_files = glob.glob(os.path.join(category_path, "*.csv"))
                
                category_plots = 0
                for csv_file in csv_files:
                    try:
                        filename = os.path.basename(csv_file).replace('.csv', '')
                        
                        # Leer datos
                        df = pd.read_csv(csv_file)
                        
                        # Determinar si usar colores de stages
                        use_stage_colors = None
                        if option == 'Option 2' and 'Segment' in df.columns:
                            # Renombrar Segment a stage para Option 2
                            df = df.rename(columns={'Segment': 'stage'})
                            use_stage_colors = stage_colors
                        
                        # Crear título
                        title = f'{option} - {split} - {category} - {filename}'
                        
                        # Crear gráfico
                        fig, (ax1, ax2) = plot_voltage_z_by_stage(
                            df,
                            vcol='Voltage',
                            zcol='Z',
                            scol='stage' if option == 'Option 2' else None,
                            stage_colors=use_stage_colors,
                            title=title
                        )
                        
                        # Guardar
                        output_file = f"{viz_category_path}/{filename}.png"
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        category_plots += 1
                        total_plots += 1
                        
                        # Mostrar progreso cada 10 archivos
                        if category_plots % 10 == 0:
                            print(f"      Procesados: {category_plots} archivos")
                        
                    except Exception as e:
                        print(f"      ERROR en {os.path.basename(csv_file)}: {e}")
                
                print(f"      Total procesados en {category}: {category_plots} archivos")
    
    print(f"\n" + "=" * 50)
    print(f"PROCESO COMPLETADO!")
    print(f"Total de visualizaciones creadas: {total_plots}")
    print(f"Estructura creada en: {viz_base}")
    
    # Mostrar estructura final
    print(f"\nESTRUCTURA CREADA:")
    print(f"{viz_base}/")
    for option in options:
        print(f"  {option}/")
        for split in splits:
            print(f"    {split}/")
            split_path = f"Data/{option}/{split}"
            if os.path.exists(split_path):
                categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
                for category in categories:
                    category_path = f"{split_path}/{category}"
                    csv_count = len(glob.glob(os.path.join(category_path, "*.csv")))
                    print(f"      {category}/ ({csv_count} archivos)")

if __name__ == "__main__":
    create_data_viz_structure()



