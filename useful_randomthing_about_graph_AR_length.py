import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# Define a class to visualize a graph
class GraphVisualization:

    # Initialize the list of edges to visualize
    def __init__(self):
        self.visual = []

    # Add an edge between nodes a and b to the list of edges
    def addEdge(self, a, b,c):
        temp = [a, b,c]
        self.visual.append(temp)
		
    # Draw the graph
    def visualize(self ,ar_edge):
        # Create a graph using the edges in the list
        G = nx.Graph()
        G.add_weighted_edges_from(self.visual)
        
        # Draw the graph in a circular layout
        node_color = [v for v in ar_edge]
        #node_size = [v*100 for v in ar_edge]

        #edgewidths = [w for ([a, b,w]) in self.visual]
        lenght = [w for ([a, b,w]) in self.visual]
        pos = nx.kamada_kawai_layout(G, weight='weight')
        edgewidths = [np.sqrt(1/abs(u-v)) for u, v in G.edges()]
        
        nx.draw(G, pos,node_color = node_color ,width = edgewidths,with_labels= True)
        #nx.draw_circular(G ,with_labels= True)

# Define a function to calculate visibility between nodes based on a given time series
def visibility(series ,num_steps, G):
    # Initialize a 2D array to store whether each pair of nodes is connected
    arr = [[0 for _ in range(0,num_steps+1)] for _ in range(0,num_steps+1)]

    # Initialize an array to store the degree of each node
    node_edge = np.zeros(num_steps)

    # Loop over all pairs of nodes to check for visibility
    for i in range(0, num_steps):
        for j in range (0,i) :
            if(i == 0) :
                break
            ch = 0
            for k in range (j,i) :
                if( series[k] > series[i] + (series[j]-series[i])*(i-k)/(i-j) ) :
                    ch =1 
                    break
            if (ch == 0 or i-1 == j) :
                # Add an edge between nodes i and j if they are visible
                G.addEdge(i+1,j+1,np.sqrt((series[i] - series[j])**2 + (i-j)**2 ))
                node_edge[i]+=1
                node_edge[j]+=1
                arr[i][j] = 1
                arr[j][i] = 1
                plt.plot((i,j),(series[i],series[j]))
                
    # Return the degree of each node
    return node_edge

# Define a function for computing the frequency distribution of edge weights
def get_freq_dist(ar_edge):
    max_val = int(round(max(ar_edge)))
    freq = np.zeros(max_val + 1)
    for i in range(len(ar_edge)):
        freq[int(round(ar_edge[i]))] += 1
    return freq

#-------------------------------------------------------------

# Set the number of time steps and the order of the AR model
num_steps = 100
porder = 4

# Import generated whitenoise 
#whitenoise = pd.read_excel('white_noise.xlsx')
#white_noise = whitenoise['col'].values.tolist()
white_noise = np.random.randn(num_steps)

# Generate the AR model coefficients
arcoeff_1 = pd.read_excel('parameter.xlsx')
arcoeff = arcoeff_1['col'].values.tolist()
# arcoeff = np.random.uniform(-1, 1, porder) #editing

# Generate the AR model time series
ar_model = np.zeros(num_steps)

for i in range(porder, num_steps):
    for j in range(porder):
        ar_model[i] += arcoeff[j] * ar_model[i-j-1]
    ar_model[i] += white_noise[i]

all_cases = num_steps - porder +1
start = 0
test = 10
trial = 1
#--------------------------------------------------------------
plt.plot(ar_model,color = 'black',linewidth = 0.5)
plt.title('AR({}) test #{}.png'.format(porder,trial))
plt.savefig('AR({})_test_#{}.png'.format(porder,trial))
# Calculate the PACF and ACF of the AR model
lags = min(10, num_steps-1)
#fig, ax = plt.subplots(2, 1, figsize=(10, 6))
#plot_pacf(ar_model, lags=lags, ax=ax[0], title='AR Partial Autocorrelation Function (PACF)')
#plot_acf(ar_model, lags=lags, ax=ax[1], title='AR Autocorrelation Function (ACF)')
#plt.tight_layout()

# Sub plots
# plt.subplots()

#-------------------------------------------------------------

# Create a new GraphVisualization object
G = GraphVisualization()

# Compute the visibility graph and return the node degrees
ar_edge = visibility(ar_model ,num_steps ,G)

# Plot the AR model time series
plt.plot(ar_model,color = 'black',linewidth = 0.5)
#plt.title('AR({}) Model Time Series'.format(porder))
#plt.xlabel('Time')
#plt.ylabel('Value')

# Create a new plot to visualize the edges in the visibility graph
#lt.subplots()
# plt.plot(ar_edge)
#plt.title('AR({}) Edges '.format(porder))
#plt.xlabel('Time')
#plt.ylabel('Value')

# Create a new plot to visualize the distribution edges in the visibility graph
#plt.subplots()
freq = get_freq_dist(ar_edge)
# plt.plot(freq)
#plt.title('AR({}) Edges Distribution'.format(porder))
#plt.xlabel('Edges (Number)')
#plt.ylabel('Edges')
plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

# Display the plots

plt.subplots()
G.visualize(ar_edge)
plt.title('AR({}) test at {} Graph #{} '.format(porder,test,trial)) #num_steps - porder +1 - all_cases
#plt.savefig('AR({})_test_at_{}_Graph_#{}.png'.format(porder,test,trial))

plt.show()
