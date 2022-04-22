import networkit as nk
import sys

def delegate_reader(weighted : bool, timestamped : bool):
    if not weighted and not timestamped:
        return nk.graphio.SNAPGraphReader(directed=True, remapNodes=True, nodeCount=0)
    else:
        return nk.graphio.EdgeListReader(' ', firstNode=1, commentPrefix='#', continuous=True, directed=True)

class DegreeDistribution:
    min_degree : int = sys.maxsize
    max_degree : int = 0
    mean_degree : float = 0.0

def compute_degree_distribution(graph : nk.Graph):
    deg_dist = DegreeDistribution()

    for n in graph.iterNodes():
        deg = graph.degree(n)
        deg_dist.min_degree = min(deg_dist.min_degree, deg)
        deg_dist.max_degree = max(deg_dist.max_degree, deg)
        deg_dist.mean_degree = deg_dist.mean_degree + deg

    deg_dist.mean_degree = deg_dist.mean_degree / graph.numberOfNodes()
    deg_dist.mean_degree = round(deg_dist.mean_degree, 1)

    return deg_dist

def main():
    if len(sys.argv) != 4:
        print('Need to pass filename, weighted = 1, timestamped = 1')
        return

    filename : str = sys.argv[1]
    weighted : bool = sys.argv[2]
    timestamped : bool = sys.argv[3]

    # reader = nk.graphio.SNAPGraphReader(directed=True, remapNodes=True, nodeCount=0)
    
    reader = nk.graphio.EdgeListReader(' ', firstNode=1, commentPrefix='#', continuous=True, directed=True)

    print('Reading graph file [' + filename + '], weighted: ' + str(weighted) + ', timestamped: ' + str(timestamped))
    graph : nk.Graph = reader.read(filename)

    deg_dist : DegreeDistribution = compute_degree_distribution(graph)

    print('deg_min: ' + str(deg_dist.min_degree))
    print('deg_max: ' + str(deg_dist.max_degree))
    print('mean_degree: ' + str(deg_dist.mean_degree))

  
if __name__=="__main__":
    main()
