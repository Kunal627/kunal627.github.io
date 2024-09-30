import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np

# Function to plot a matrix
def plot_matrix(matrix, tok2pos):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(cax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]}', va='center', ha='center', color='black')

    # Set axis labels and title
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=[x for x in tok2pos.keys()])
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