## Train a MB-GAN based on custom microbiome CSV data
from model import MBGAN
from scipy.stats import describe
from utils import *
import pandas as pd
import numpy as np

NAME = "mbgan_custom"
EXP_DIR = "custom_microbiome"
CSV_FILE = "./course_data/microbiome_relative_abundance.csv"
ITERS = 10000
BATCH_SIZE = 32
SAVE_INTERVAL = 1000


def get_save_fn(taxa_list):
    def fn(model, epoch):
        table_dir = os.path.join(model.log_dir, "tables")
        if not os.path.exists(table_dir):
            os.makedirs(table_dir)
        
        res = model.predict(1000, transform=None, seed=None)
        sparsity, entropy = get_sparsity(res), shannon_entropy(res)
        print("sparsity: %s" % str(describe(sparsity)))
        print("entropy: %s" % str(describe(entropy)))
        
        filename = "{:s}_{:06d}--{:.4f}--{:.4f}.csv".format(
            model.model_name, epoch, np.mean(sparsity), np.mean(entropy))
        
        pd.DataFrame(res, columns=taxa_list).to_csv(os.path.join(table_dir, filename))
        
    return fn


def load_custom_csv_data(csv_file):
    """Load custom CSV file with SampleID and taxa abundance columns"""
    # Read the CSV file
    data = pd.read_csv(csv_file, index_col=0)  # Use first column (SampleID) as index
    
    # The data is already in the right format: samples as rows, taxa as columns
    # Convert to numpy array and normalize (data should already be relative abundances)
    data_array = data.values
    taxa_list = data.columns.tolist()
    
    print(f"Loaded data shape: {data_array.shape}")
    print(f"Number of samples: {data_array.shape[0]}")
    print(f"Number of taxa: {data_array.shape[1]}")
    print(f"Data range: {data_array.min():.6f} to {data_array.max():.6f}")
    print(f"Sample sparsity statistics: {describe(get_sparsity(data_array))}")
    print(f"Sample entropy statistics: {describe(shannon_entropy(data_array))}")
    
    return data_array, taxa_list


def create_phylo_matrix(taxa_list):
    """Create a simple phylogenetic matrix for the taxa"""
    # For this example, we'll create a simple identity matrix
    # In a real scenario, you might want to use actual phylogenetic relationships
    n_taxa = len(taxa_list)
    
    # Create adjacency matrix (each taxa connected to itself for simplicity)
    adj_matrix = [(i, i) for i in range(n_taxa)]
    
    # Create indices mapping
    taxa_indices = {i: taxa for i, taxa in enumerate(taxa_list)}
    
    return adj_matrix, taxa_indices


if __name__ == '__main__':
    # Load the custom CSV dataset
    print("Loading custom CSV data...")
    data_array, taxa_list = load_custom_csv_data(CSV_FILE)
    
    # Create phylogenetic matrix
    print("Creating phylogenetic matrix...")
    adj_matrix, taxa_indices = create_phylo_matrix(taxa_list)
    tf_matrix = adjmatrix_to_dense(adj_matrix, shape=(len(taxa_list), len(taxa_indices)))
    
    # Model configuration
    model_config = {
        'ntaxa': len(taxa_list),  # Use actual number of taxa
        'latent_dim': 100,
        'generator': {'n_channels': 512},
        'critic': {'n_channels': 256, 'dropout_rate': 0.25, 
                   'tf_matrix': tf_matrix, 't_pow': 1000.}
    }
    
    train_config = {
        'generator': {'optimizer': ('RMSprop', {}), 'lr': 0.00005},
        'critic': {'loss_weights': [1, 1, 10], 
                   'optimizer': ('RMSprop', {}), 'lr': 0.00005},
    }
    
    # Train the model
    print("Starting MB-GAN training...")
    print(f"Training samples: {data_array.shape[0]}")
    print(f"Number of taxa: {len(taxa_list)}")
    print(f"Training iterations: {ITERS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    mbgan = MBGAN(NAME, model_config, train_config)
    mbgan.train(data_array, iteration=ITERS, batch_size=BATCH_SIZE, 
                n_critic=5, n_generator=1, save_fn=get_save_fn(taxa_list), 
                save_interval=SAVE_INTERVAL, experiment_dir=EXP_DIR,
                pre_processor=None, verbose=0)
    
    print("Training completed!")
