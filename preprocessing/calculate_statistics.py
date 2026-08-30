#claude script 
def calculate_stats(entries):
    """
    Υπολογίζει στατιστικά μήκους (πλήθος frames) πάνω σε μια λίστα valid entries.
    entries: λίστα από (npy_path, label)
    """
    lengths = []
    for npy_path, label in entries:
        landmarks = np.load(npy_path)
        lengths.append(landmarks.shape[0])

    lengths = np.array(lengths)

    print(f"Σύνολο glosses: {len(lengths)}")
    print(f"Mean: {lengths.mean():.1f}")
    print(f"Median: {np.median(lengths):.1f}")
    print(f"Min: {lengths.min()}")
    print(f"Max: {lengths.max()}")
    print(f"95th percentile: {np.percentile(lengths, 95):.1f}")





