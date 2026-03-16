import matplotlib.pyplot as plt

def plot_search_trace(title, feature_sets, accuracies, filename, y_min=0, y_max=100):
    """Generates a bar chart formatted similarly to the CS170 sample report."""
    plt.figure(figsize=(12, 5)) # Slightly wider to accommodate more bars
    
    bars = plt.bar(feature_sets, accuracies, color='gray')
    
    # format graph
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Current Feature Set', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(y_min, y_max)
    
    # if bar is 0 (used for the "omitted" gap), remove its x-tick label 
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rotate x-axis labels if there are a lot of bars so they don't overlap
    if len(feature_sets) > 6:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    # --- 1. Small Dataset: Forward Selection ---
    # Prepended {} baseline
    small_fwd_x = ['{}', '{15}', '{15, 1}', '{15, 1, 2}', '{15, 1, 2, 6}']
    small_fwd_y = [73.6, 86.0, 97.8, 94.2, 89.6]
    plot_search_trace('Small Dataset: Forward Selection', small_fwd_x, small_fwd_y, 'fig1_small_forward.png')

    # --- 2. Small Dataset: Backward Elimination ---
    # Prepended {All 16} baseline. 
    # Abbreviated subsets with > 4 features to prevent x-axis text overlap.
    small_bwd_x = [
        '{All 16}', '{15 feats}', '{14 feats}', '{13 feats}', '{12 feats}', 
        '{11 feats}', '{10 feats}', '{9 feats}', '{8 feats}', '{7 feats}', 
        '{6 feats}', '{5 feats}', '{1, 3, 8, 15}', '{1, 8, 15}', '{1, 15}', '{15}'
    ]
    small_bwd_y = [
        73.6, 74.6, 75.8, 77.0, 77.4, 
        79.8, 82.6, 82.6, 81.2, 83.2, 
        83.4, 88.2, 90.0, 92.2, 97.8, 86.0
    ]
    plot_search_trace('Small Dataset: Backward Elimination', small_bwd_x, small_bwd_y, 'fig2_small_backward.png')

    # --- 3. Large Dataset: Forward Selection ---
    # Prepended {} baseline. Values rounded to one decimal place for a clean y-axis.
    large_fwd_x = ['{}', '{48}', '{48, 46}', '{48, 46, 39}', '{48, 46, 39, 51}']
    large_fwd_y = [69.0, 82.9, 97.0, 96.8, 94.7]
    plot_search_trace('Large Dataset: Forward Selection', large_fwd_x, large_fwd_y, 'fig3_large_forward.png')
    
    # --- 4. Large Dataset: Backward Elimination ---
    # Prepended {All 64} baseline. 0 is used to create the visual "omitted" gap.
    large_bwd_x = [
        '{All 64}', '{63 feats}', 'omitted feats 62 to 53 for space', 
        '{52 feats}', '{51 feats}', '{50 feats}', 
        '{49 feats}', '{48 feats (Best)}', '{47 feats}', '{46 feats}'
    ]
    large_bwd_y = [
        69.0, 69.5, 0, 
        73.2, 73.6, 73.4, 
        73.8, 74.3, 74.1, 73.9
    ]
    plot_search_trace('Large Dataset: Backward Elimination', large_bwd_x, large_bwd_y, 'fig4_large_backward.png')

    # --- 5. Extra Credit: Dry Bean Dataset (Forward Selection) ---
    extra_fwd_x = [
        '{}', '{2}', '{2, [X]}', '...', 
        '{[BEST_SUBSET_MINUS_1]}', '{[BEST_SUBSET]}', '{[NEXT_SUBSET]}'
    ]
    extra_fwd_y = [
        14.2, 55.4, 0.0, 0,
        0.0, 0.0, 0.0 # Placeholder values, replace with real values when available.
    ]
    plot_search_trace('Extra Credit: Dry Bean Dataset (Forward)', extra_fwd_x, extra_fwd_y, 'fig5_extra_credit.png')