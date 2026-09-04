import case_reader as cr
train_files = ['/ceph/abal/QML/JetClass/VQC/merged/flat_train/TTBar+ZJets_flat.h5']
input_n = 10
train_n = 100
batch_size = 10
train_loader = cr.OneP1QDataLoader(
    filelist=train_files,
    batch_size=batch_size,
    input_shape=(input_n, 3),
    train=True,
    max_samples=train_n,
    normalize_pt=True,
    use_subjet_PFCands=False,
    logger=None,
    dataset="jetclass",
    selection="random",
)

for data,labels in train_loader:
    print(data[:,0,2])  # This will display the batch data
    print(labels)