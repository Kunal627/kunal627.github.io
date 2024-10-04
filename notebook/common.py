import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np

# Function to plot a matrix
def plot_matrix(matrix, xlabels):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='Blues')
    #fig.colorbar(cax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]}', va='center', ha='center', color='black')

    # Set axis labels and title
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=[x for x in xlabels])
    plt.title("Vocab Embedding Matrix")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Row")

    return plt

def get_plot(matrix, pos2token, inp_seq):
    # Select color for each point
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'magenta', 'brown']
    colors = colors[:matrix.shape[0]]
    plt.figure(figsize=(6, 6)) 
    i = 0
    # Plot the arrows from the origin (0, 0) to each point
    for (x, y), color, token in zip(matrix, colors, inp_seq):
        plt.arrow(0, 0, x, y, head_width=0.03, head_length=0.04, fc=color, ec=color)
        plt.text(x + 0.035, y + 0.035, f'(pos={i} {token})', fontsize=9, color=color)
        i += 1

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot final embeddings in the input sequence sequence')
    return plt

def get_encoding(embedding_matrix, inp_seq, tok2pos):
    encoding = {}
    for i, token in enumerate(inp_seq):
        encoding[token] = embedding_matrix[tok2pos[token]]
    return encoding

def positional_encoding(seq_len, n_embed, theta):
    seq_pe = []
    for pos in range(seq_len):
        pe = []
        for i in range(n_embed // 2):
            pe1 = math.sin(pos/(theta ** (2 * i / n_embed)))
            pe2 = math.cos(pos/(theta ** (2 * i / n_embed)))
            pe.extend([pe1,pe2])
        seq_pe.append(pe)
            
    return np.array(seq_pe).round(3)

def cosine_similarity(token1, token2):
    """
    Computes cosine similarity between two tokens.
    
    Parameters:
    - token1: First token (embedding vector)
    - token2: Second token (embedding vector)
    
    Returns:
    - similarity: Cosine similarity between the two tokens
    """
    dot_product = np.dot(token1, token2)
    norm1 = np.linalg.norm(token1)
    norm2 = np.linalg.norm(token2)
    return dot_product / (norm1 * norm2)

def euclidean_distance(token1, token2):
    """
    Computes Euclidean distance between two tokens.
    
    Parameters:
    - token1: First token (embedding vector)
    - token2: Second token (embedding vector)
    
    Returns:
    - Distance: Euclidean distance between the two tokens
    """
    return np.linalg.norm(token1 - token2)


def get_rope_encoding(x, n_embed, seq_len, scale=10000.0):
    pos = np.arange(seq_len)  # (seq_len, ) get the token position (m) in the sequence
    freq = 1.0 /(scale ** (np.arange(0, n_embed, step=2)/ n_embed ))[: n_embed // 2] # (n_embed // 2, ) get the frequency for each dimension
    thetas = np.einsum('i,j->ij', pos, freq)  # (seq_len, n_embed // 2) get the angle for each token position and each dimension
    cosx = np.cos(thetas)  # (seq_len, n_embed // 2) get the cos for each token position and each dimension
    sinx = np.sin(thetas)  # (seq_len, n_embed // 2) get the sin for each token position and each dimension
    x = x.reshape(seq_len, n_embed//2, 2)  # (seq_len, n_embed // 2, 2) reshape the input to get the x and y for each token position and each dimension
    thetas = np.concatenate([cosx, sinx], axis=-1)  # (seq_len, n_embed) concatenate cos and sin to get the final thetas
    rope = np.stack([x[..., 0] * cosx - x[..., 1] * sinx, x[..., 0] * sinx + x[..., 1] * cosx], axis=-1)  # (seq_len, n_embed, 2) get the rope encoding
    return rope