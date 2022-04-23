import networkit as nk
import sys

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
    if len(sys.argv) != 3:
        print('Need to pass filename and config id.')
        return

    filename : str = sys.argv[1]
    config_id : int = sys.argv[2]

    reader = {
      1 : nk.graphio.SNAPGraphReader(directed=True, remapNodes=True, nodeCount=0),
      2 : nk.graphio.EdgeListReader(' ', firstNode=0, commentPrefix='%', continuous=True, directed=True),
      3 : nk.graphio.EdgeListReader(' ', firstNode=0, commentPrefix='#', continuous=True, directed=True),
      4 : nk.graphio.KONECTGraphReader(remapNodes=False, handlingmethod=nk.graphio.MultipleEdgesHandling.DiscardEdges)
    }.get(config_id, nk.graphio.SNAPGraphReader(directed=True, remapNodes=True, nodeCount=0))

    print('Reading graph file [' + filename + '], config id: ' + str(config_id))
    graph : nk.Graph = reader.read(filename)

    deg_dist : DegreeDistribution = compute_degree_distribution(graph)

    print('deg_min: ' + str(deg_dist.min_degree))
    print('deg_max: ' + str(deg_dist.max_degree))
    print('mean_degree: ' + str(deg_dist.mean_degree))

  
if __name__=="__main__":
    main()
