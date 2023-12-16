from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

   
def visualization (source_data, source_labels, target_data, target_labels, analysis):
  # PCA/TSNE code adapted from https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
  
  colour_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(target_data)])
  idx = np.random.permutation(len(target_data))[:anal_sample_no]
    
  # Data preprocessing
  source_data = np.asarray(source_data)
  target_data = np.asarray(target_data) 
  source_labels = np.asarray(source_labels)
  target_labels = np.asarray(target_labels)
  
  source_data = source_data[idx]
  target_data = target_data[idx]
  source_labels = source_labels[idx]
  target_labels = target_labels[idx]
  
  # Visualization parameter        
  colours = [colour_list[source_labels[i]] for i in range(anal_sample_no)] + [colour_list[target_labels[i]] for i in range(anal_sample_no)]

  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(source_data)
    pca_results = pca.transform(source_data)
    pca_hat_results = pca.transform(target_data)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colours[:anal_sample_no],
                marker = "x", alpha = 0.8, label = "Source Distribution")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colours[anal_sample_no:2*anal_sample_no],
                marker = "o", alpha = 0.8, label = "Target Distribution")
    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x_pca')
    plt.ylabel('y_pca')
    plt.show()

  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((source_data, target_data), axis = 0)
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colours[:anal_sample_no], marker = "x", alpha = 0.8, label = "Source Distribution")
    plt.scatter(tsne_results[anal_sample_no:2*anal_sample_no,0], tsne_results[anal_sample_no:2*anal_sample_no,1], 
                c = colours[anal_sample_no:2*anal_sample_no], marker = "o", alpha = 0.8, label = "Target Distribution")
    ax.legend()
    
    plt.title('t-SNE plot')
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')
    plt.show()
    