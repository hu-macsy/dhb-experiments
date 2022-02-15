/*
 * Graph.cpp
 *
 *  Created on: 01.06.2014
 *      Author: Christian Staudt
 *              Klara Reichard <klara.reichard@gmail.com>
 *              Marvin Ritter <marvin.ritter@gmail.com>
 */

#include <cmath>
#include <map>
#include <random>
#include <sstream>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

namespace NetworKit {

static std::shared_ptr<dhb::BlockManager>
    globalBlockManager(std::make_shared<dhb::BlockManager>(sizeof(dhb::Target)));

/** CONSTRUCTORS **/

Graph::Graph(count n, bool weighted, bool directed)
    : n(n), m(0), storedNumberOfSelfLoops(0), z(n), omega(0), t(0),

      weighted(weighted),  // indicates whether the graph is weighted or not
      directed(directed),  // indicates whether the graph is directed or not
      edgesIndexed(false), // edges are not indexed by default

      exists(n, true), outMatrix(globalBlockManager), inMatrix(globalBlockManager) {
    outMatrix.resize(n);
    if (directed)
        inMatrix.resize(n);
}

Graph::Graph(std::initializer_list<WeightedEdge> edges) : Graph(0, true) {
    using namespace std;

    /* Number of nodes = highest node index + 1 */
    for (const auto &edge : edges) {
        node x = max(edge.u, edge.v);
        while (numberOfNodes() <= x) {
            addNode();
        }
    }

    /* Now add all of the edges */
    for (const auto &edge : edges) {
        addEdge(edge.u, edge.v, edge.weight);
    }
}

Graph::Graph(const Graph &G, bool weighted, bool directed)
    : n(G.n), m(G.m), storedNumberOfSelfLoops(G.storedNumberOfSelfLoops), z(G.z), omega(0), t(G.t),
      weighted(weighted), directed(directed),
      edgesIndexed(false), // edges are not indexed by default
      exists(G.exists), outMatrix(globalBlockManager), inMatrix(globalBlockManager) {
    if (G.isDirected() == directed) {
        outMatrix = G.outMatrix;
        inMatrix = G.inMatrix;
    } else {
        if (G.isDirected()) {
            assert(!directed);

            // We copy both G.outMatrix and G.inMatrix into our outMatrix.
            outMatrix.resize(z);
            G.outMatrix.for_edges([&](node u, node v, auto ed) {
                outMatrix.insert(u, v, {ed.weight, ed.id}, false);
            });
            G.inMatrix.for_edges([&](node u, node v, auto ed) {
                outMatrix.insert(u, v, {ed.weight, ed.id}, false);
            });
        } else {
            assert(!G.isDirected());
            assert(directed);

            // We copy G.outMatrix into both our outMatrix and our inMatrix.
            outMatrix.resize(z);
            inMatrix.resize(z);
            G.outMatrix.for_edges([&](node u, node v, auto ed) {
                outMatrix.insert(u, v, {ed.weight, ed.id}, false);
                inMatrix.insert(u, v, {ed.weight, ed.id}, false);
            });
        }
    }
}

void Graph::preallocateUndirected(node u, size_t size) {
    INFO("preallocateUndirected() is ignored");
    /*
        assert(!directed);
        assert(exists[u]);
        outEdges[u].reserve(size);
        if (weighted) {
            outEdgeWeights[u].reserve(size);
        }
        if (edgesIndexed) {
            outEdgeIds[u].reserve(size);
        }
    */
}

void Graph::preallocateDirected(node u, size_t outSize, size_t inSize) {
    preallocateDirectedOutEdges(u, outSize);
    preallocateDirectedInEdges(u, inSize);
}

void Graph::preallocateDirectedOutEdges(node u, size_t outSize) {
    INFO("preallocateDirectedOutEdges() is ignored");
    /*
        assert(directed);
        assert(exists[u]);
        outEdges[u].reserve(outSize);

        if (weighted) {
            outEdgeWeights[u].reserve(outSize);
        }
        if (edgesIndexed) {
            outEdges[u].reserve(outSize);
        }
    */
}

void Graph::preallocateDirectedInEdges(node u, size_t inSize) {
    INFO("preallocateDirectedInEdges() is ignored");
    /*
        assert(directed);
        assert(exists[u]);
        inEdges[u].reserve(inSize);

        if (weighted) {
            inEdgeWeights[u].reserve(inSize);
        }
        if (edgesIndexed) {
            inEdgeIds[u].reserve(inSize);
        }
    */
}
/** PRIVATE HELPERS **/

index Graph::indexInInEdgeArray(node v, node u) const {
    throw std::runtime_error("Fix indexInInEdgeArray");
    /*
        if (!directed) {
            return indexInOutEdgeArray(v, u);
        }
        for (index i = 0; i < inEdges[v].size(); i++) {
            node x = inEdges[v][i];
            if (x == u) {
                return i;
            }
        }
        return none;
    */
}

index Graph::indexInOutEdgeArray(node u, node v) const {
    throw std::runtime_error("Fix indexInOutEdgeArray");
    /*
        for (index i = 0; i < outEdges[u].size(); i++) {
            node x = outEdges[u][i];
            if (x == v) {
                return i;
            }
        }
        return none;
    */
}

/** EDGE IDS **/

void Graph::indexEdges(bool force) {
    if (edgesIndexed && !force)
        return;

    omega = 0; // reset edge ids (for re-indexing)

    if (!directed) {
        // Assign one direction.
        forNodes([&](node u) {
            for (auto e : outMatrix.neighbors(u)) {
                if (u >= e.vertex()) {
                    e.data().id = omega++;
                } else {
                    e.data().id = none;
                }
            }
        });

        // Assign other direction.
        balancedParallelForNodes([&](node u) {
            for (auto e : outMatrix.neighbors(u)) {
                if (u >= e.vertex())
                    continue;
                assert(e.id == none);
                auto nv = outMatrix.neighbors(e.vertex());
                auto it = nv.iterator_to(u);
                assert(it != nv.end());
                assert(it->data().id != none);
                e.data().id = it->data().id;
            }
        });
    } else {
        // Assign out edges.
        forNodes([&](node u) {
            for (auto e : outMatrix.neighbors(u))
                e.data().id = omega++;
        });

        // Assign in edges.
        balancedParallelForNodes([&](node u) {
            for (auto e : inMatrix.neighbors(u)) {
                auto nv = outMatrix.neighbors(e.vertex());
                auto it = nv.iterator_to(u);
                assert(it != nv.end());
                assert(it->data().id != none);
                e.data().id = it->data().id;
            }
        });
    }

    edgesIndexed = true; // remember that edges have been indexed so that addEdge
                         // needs to create edge ids
}

edgeid Graph::edgeId(node u, node v) const {
    if (!edgesIndexed)
        throw std::runtime_error("edges have not been indexed - call indexEdges first");

    auto nu = outMatrix.neighbors(u);
    auto it = nu.iterator_to(v);
    if (it == nu.end())
        throw std::runtime_error("Edge does not exist");
    return it->data().id;
}

/** GRAPH INFORMATION **/

void Graph::shrinkToFit() {
    INFO("shrinkToFit() is ignored");
}

void Graph::compactEdges() {
    // This is a no-op for now.
}

void Graph::sortEdges() {
    auto less = [](auto v, auto w) { return v.vertex < w.vertex; };
    if (directed) {
        parallelForNodes([&](node u) {
            outMatrix.sort(u, less);
            inMatrix.sort(u, less);
        });
    } else {
        parallelForNodes([&](node u) { outMatrix.sort(u, less); });
    }
}

edgeweight Graph::computeWeightedDegree(node u, bool inDegree, bool countSelfLoopsTwice) const {
    if (weighted) {
        edgeweight sum = 0.0;
        auto sumWeights = [&](node v, edgeweight w) {
            sum += (countSelfLoopsTwice && u == v) ? 2. * w : w;
        };
        if (inDegree) {
            forInNeighborsOf(u, sumWeights);
        } else {
            forNeighborsOf(u, sumWeights);
        }
        return sum;
    }

    count sum = inDegree ? degreeIn(u) : degreeOut(u);
    auto countSelfLoops = [&](node v) { sum += (u == v); };

    if (countSelfLoopsTwice && numberOfSelfLoops()) {
        if (inDegree) {
            forInNeighborsOf(u, countSelfLoops);
        } else {
            forNeighborsOf(u, countSelfLoops);
        }
    }

    return static_cast<edgeweight>(sum);
}

/** NODE MODIFIERS **/

node Graph::addNode() {
    node v = z; // node gets maximum id
    z++;        // increment node range
    n++;        // increment node count

    // update per node data structures
    exists.push_back(true);

    outMatrix.resize(v + 1);
    if (directed)
        inMatrix.resize(v + 1);

    return v;
}

node Graph::addNodes(count numberOfNewNodes) {
    if (numberOfNewNodes < 10) {
        // benchmarks suggested, it's cheaper to call 10 time emplace_back than resizing.
        while (numberOfNewNodes--)
            addNode();

        return z - 1;
    }

    z += numberOfNewNodes;
    n += numberOfNewNodes;

    // update per node data structures
    exists.resize(z, true);

    outMatrix.resize(z);
    if (directed)
        inMatrix.resize(z);

    return z - 1;
}

void Graph::removeNode(node v) {
    assert(v < z);
    assert(exists[v]);

    // Remove all outgoing and ingoing edges
    while (degreeOut(v))
        removeEdge(v, getIthNeighbor(v, 0));
    if (isDirected())
        while (degreeIn(v))
            removeEdge(getIthInNeighbor(v, 0), v);

    exists[v] = false;
    n--;
}

void Graph::restoreNode(node v) {
    assert(v < z);
    assert(!exists[v]);

    exists[v] = true;
    n++;
}

/** NODE PROPERTIES **/

edgeweight Graph::weightedDegree(node u, bool countSelfLoopsTwice) const {
    return computeWeightedDegree(u, false, countSelfLoopsTwice);
}

edgeweight Graph::weightedDegreeIn(node u, bool countSelfLoopsTwice) const {
    return computeWeightedDegree(u, true, countSelfLoopsTwice);
}

/** EDGE MODIFIERS **/

void Graph::addEdge(node u, node v, edgeweight ew) {
    assert(u < z);
    assert(exists[u]);
    assert(v < z);
    assert(exists[v]);

    // If edges indexed, give new ID.
    edgeid id = none;
    if (edgesIndexed)
        id = omega;

    if (!outMatrix.insert(u, v, {ew, id}, false))
        return;

    // Increase number of edges.
    ++m;
    if (edgesIndexed)
        ++omega;
    if (u == v) // Count self loop.
        ++storedNumberOfSelfLoops;

    if (directed) {
        inMatrix.insert(v, u, {ew, id}, false);
    } else {
        if (u != v)
            outMatrix.insert(v, u, {ew, id}, false);
    }
}
void Graph::addPartialEdge(Unsafe, node u, node v, edgeweight ew, uint64_t index) {
    assert(!directed);
    assert(u < z);
    assert(exists[u]);
    assert(v < z);
    assert(exists[v]);

    outMatrix.insert(u, v, {ew, index}, false);
}
void Graph::addPartialOutEdge(Unsafe, node u, node v, edgeweight ew, uint64_t index) {
    assert(directed);
    assert(u < z);
    assert(exists[u]);
    assert(v < z);
    assert(exists[v]);

    outMatrix.insert(u, v, {ew, index}, false);
}
void Graph::addPartialInEdge(Unsafe, node u, node v, edgeweight ew, uint64_t index) {
    assert(directed);
    assert(u < z);
    assert(exists[u]);
    assert(v < z);
    assert(exists[v]);

    inMatrix.insert(u, v, {ew, index}, false);
}

template <typename T>
void erase(node u, index idx, std::vector<std::vector<T>> &vec) {
    vec[u][idx] = vec[u].back();
    vec[u].pop_back();
}

void Graph::removeEdge(node u, node v) {
    assert(u < z);
    assert(exists[u]);
    assert(v < z);
    assert(exists[v]);

    if (!hasEdge(u, v)) {
        std::stringstream strm;
        strm << "edge (" << u << "," << v << ") does not exist";
        throw std::runtime_error(strm.str());
    }

    m--; // decrease number of edges
    if (u == v)
        storedNumberOfSelfLoops--;

    outMatrix.removeEdge(u, v);

    if (directed) {
        inMatrix.removeEdge(v, u);
    } else {
        if (u != v)
            outMatrix.removeEdge(v, u);
    }
}

void Graph::removeAllEdges() {
    parallelForNodes([&](const node u) {
        removePartialOutEdges(unsafe, u);
        if (isDirected()) {
            removePartialInEdges(unsafe, u);
        }
    });

    m = 0;
}

void Graph::removeSelfLoops() {
    INFO("storedNumberOfSelfLoops: ", storedNumberOfSelfLoops);
    forNodes([&](const node u) {
        if (!hasEdge(u, u))
            return;
        removeEdge(u, u);
    });

    INFO("storedNumberOfSelfLoops: ", storedNumberOfSelfLoops);
    assert(!storedNumberOfSelfLoops);
}

void Graph::removeMultiEdges() {
    // No-op because we don't add multi-edges in the first place.
}

void Graph::swapEdge(node s1, node t1, node s2, node t2) {
    throw std::runtime_error("Fix swapEdge");
    /*
        index s1t1 = indexInOutEdgeArray(s1, t1);
        if (s1t1 == none)
            throw std::runtime_error("The first edge does not exist");
        index t1s1 = indexInInEdgeArray(t1, s1);

        index s2t2 = indexInOutEdgeArray(s2, t2);
        if (s2t2 == none)
            throw std::runtime_error("The second edge does not exist");
        index t2s2 = indexInInEdgeArray(t2, s2);

        std::swap(outEdges[s1][s1t1], outEdges[s2][s2t2]);

        if (directed) {
            std::swap(inEdges[t1][t1s1], inEdges[t2][t2s2]);

            if (weighted) {
                std::swap(inEdgeWeights[t1][t1s1], inEdgeWeights[t2][t2s2]);
            }

            if (edgesIndexed) {
                std::swap(inEdgeIds[t1][t1s1], inEdgeIds[t2][t2s2]);
            }
        } else {
            std::swap(outEdges[t1][t1s1], outEdges[t2][t2s2]);

            if (weighted) {
                std::swap(outEdgeWeights[t1][t1s1], outEdgeWeights[t2][t2s2]);
            }

            if (edgesIndexed) {
                std::swap(outEdgeIds[t1][t1s1], outEdgeIds[t2][t2s2]);
            }
        }
    */
}

bool Graph::hasEdge(node u, node v) const noexcept {
    if (u >= z || v >= z) {
        return false;
    }
    return outMatrix.neighbors(u).exists(v);
}

/** EDGE ATTRIBUTES **/

edgeweight Graph::weight(node u, node v) const {
    auto nu = outMatrix.neighbors(u);
    auto it = nu.iterator_to(v);
    if (it == nu.end())
        return nullWeight;
    return weighted ? it->data().weight : defaultEdgeWeight;
}

void Graph::setWeight(node u, node v, edgeweight ew) {
    if (!weighted) {
        throw std::runtime_error("Cannot set edge weight in unweighted graph.");
    }

    auto nu = outMatrix.neighbors(u);
    auto uIt = nu.iterator_to(v);
    if (uIt == nu.end()) {
        addEdge(u, v, ew);
        return;
    }
    uIt->data().weight = ew;

    if (directed) {
        auto nv = inMatrix.neighbors(v);
        auto vIt = nv.iterator_to(u);
        assert(vIt != nv.end());
        vIt->data().weight = ew;
    } else {
        if (u != v) {
            auto nv = outMatrix.neighbors(v);
            auto vIt = nv.iterator_to(u);
            assert(vIt != nv.end());
            vIt->data().weight = ew;
        }
    }
}

void Graph::increaseWeight(node u, node v, edgeweight ew) {
    if (!weighted) {
        throw std::runtime_error("Cannot increase edge weight in unweighted graph.");
    }

    auto nu = outMatrix.neighbors(u);
    auto uIt = nu.iterator_to(v);
    if (uIt == nu.end()) {
        addEdge(u, v, ew);
        return;
    }
    uIt->data().weight += ew;

    if (directed) {
        auto nv = inMatrix.neighbors(v);
        auto vIt = nv.iterator_to(u);
        assert(vIt != nv.end());
        vIt->data().weight += ew;
    } else {
        if (u != v) {
            auto nv = outMatrix.neighbors(v);
            auto vIt = nv.iterator_to(u);
            assert(vIt != nv.end());
            vIt->data().weight += ew;
        }
    }
}

void Graph::setWeightAtIthNeighbor(Unsafe, node u, index i, edgeweight ew) {
    throw std::runtime_error("Fix setWeightAtIthNeighbor");
}

/** SUMS **/

edgeweight Graph::totalEdgeWeight() const noexcept {
    if (weighted) {
        edgeweight sum = 0.0;
        forEdges([&](node, node, edgeweight ew) { sum += ew; });
        return sum;
    } else {
        return numberOfEdges() * defaultEdgeWeight;
    }
}

bool Graph::checkConsistency() const {
    // No-op since we do not add multi-edges in the first place.
    return true;
}

} /* namespace NetworKit */
