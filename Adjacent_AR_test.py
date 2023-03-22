import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

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
        #pos = nx.kamada_kawai_layout(G, weight='weight')
        edgewidths = [np.sqrt(1/abs(u-v)) for u, v in G.edges()]
        
        nx.draw_circular(G,node_color = node_color ,width = edgewidths,with_labels= True)
        #nx.draw_circular(G ,with_labels= True)

# Define a function to calculate visibility between nodes based on a given time series
def visibility(series ,num_steps, G):
    # Initialize a 2D array to store whether each pair of nodes is connected
    arr = [[0 for _ in range(0,num_steps+1)] for _ in range(0,num_steps+1)]

    # Initialize an array to store the degree of each node
    node_edge = np.zeros(num_steps)

    # Loop over all pairs of nodes to check for visibility
    for i in range (0, num_steps):
        for j in range (0,i) :
            if(i == 0) :
                break
            ch = 0
            for k in range (j,i) :
                if( k >= num_steps or k < 0) :
                    break
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
num_steps = 30
porder = 1

# Import generated whitenoise 
#whitenoise = pd.read_excel('white_noise.xlsx')
#white_noise = whitenoise['col'].values.tolist()
white_noise = np.random.randn(num_steps)

# Generate the AR model coefficients
arcoeff_1 = pd.read_excel('parameter.xlsx')
arcoeff = arcoeff_1['col'].values.tolist()

# Generate the AR model time series
ar_model = np.zeros(num_steps)

for i in range(porder, num_steps):
    for j in range(porder):
        ar_model[i] += arcoeff[j] * ar_model[i-j-1]
    ar_model[i] += white_noise[i]


plt.plot(ar_model,color = 'black',linewidth = 0.5)
plt.savefig('AR({})_test.png'.format(porder))
#break

for i in range (1,num_steps) :
    for j in range (0,i) :
        #all_cases-=1
        ar_sub = np.zeros(i-j+1)
        for k in range(j,i+1) : #chamge
            ar_sub[k-j] = ar_model[k]
        
        test = i-j+1 
        start = i
        #-------------------------------------------------------------
        # Calculate the PACF and ACF of the AR model
        #lags = min(10, int((porder)/2))
        #fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        #plot_pacf(ar_sub, lags=lags, ax=ax[0], title='AR Partial Autocorrelation Function (PACF)')
        #plot_acf(ar_sub, lags=lags, ax=ax[1], title='AR Autocorrelation Function (ACF)')
        #plt.tight_layout()

        # Sub plots
        plt.subplots()

        #-------------------------------------------------------------

        # Create a new GraphVisualization object
        G = GraphVisualization()

        # Compute the visibility graph and return the node degrees
        ar_edge = visibility(ar_sub ,test ,G)

        # Plot the AR model time series
        plt.plot(ar_sub,color = 'black',linewidth = 0.5)
        plt.title('AR({}) test from {}, {} lags Time series'.format(porder,i+1, test-1))
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig('AR({})_test_from_{},_{}_lags_Time_series'.format(porder,i+1, test-1))


        # Create a new plot to visualize the edges in the visibility graph
        #plt.subplots()
        #plt.plot(ar_edge)
        #plt.title('AR({})_test_from_{},_{}_lags_Edges'.format(porder,i, test))
        #plt.xlabel('Time')
        #plt.ylabel('Value')
        #plt.savefig('AR({})_test_from_{},_{}_lags_Edges'.format(porder,i, test))

        # Create a new plot to visualize the distribution edges in the visibility graph
        #plt.subplots()
        #freq = get_freq_dist(ar_edge)
        #plt.plot(freq)
        #plt.title('AR({})_test_from_{},_{}_lags_Edges distribution'.format(porder,i, test))
        #plt.xlabel('Edges (Number)')
        #plt.ylabel('Edges')
        #plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        #plt.gca().yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        #plt.savefig('AR({})_test_at_{}_Edges distribution_#{}.png'.format(porder,start, test-1))

        plt.subplots()
        G.visualize(ar_edge)
        plt.title('AR({}) test from {}, {} lags Visibility Graph'.format(porder,i+1, test-1))
        plt.savefig('AR({})_test_from_{},_{}_lags_Visibility_Graph'.format(porder,i+1, test-1))

        # Display the plots
        plt.plot()