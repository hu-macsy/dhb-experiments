#include <dhb/graph.h>

#include <gdsb/experiment.h>
#include <gdsb/graph.h>
#include <gdsb/graph_io.h>
#include <gdsb/timer.h>

#include <networkit/graph/GraphBuilder.hpp>

#include "lib/CLI11.hpp"

#include <omp.h>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"nk_measurements"};

    std::string experiment_name = "Unkown";
    app.add_option("-e, --experiment", experiment_name, "Specify the name of the experiment");

    std::string graph_raw_path = "graphs/foodweb-baydry.konect";
    app.add_option("-g,--graph", graph_raw_path, "Full path to graph file.");

    std::string insertion_routine{"none"};
    app.add_option("-r,--insertion-routine", insertion_routine, "Insertion routine to be used.");

    unsigned int omp_thread_count = 1;
    app.add_option("-t,--thread-count", omp_thread_count,
                   "# of threads to execute OMP regions with.");

    bool weighted_graph_file = false;
    app.add_flag("-w, --weighted", weighted_graph_file, "Graph file contains weight.");

    bool temporal_graph = false;
    app.add_flag("-y,--temporal-graph", temporal_graph,
                 "Flag in case the graph file contains timestamps."
                 "This will sort all edges based on the time stamp.");

    CLI11_PARSE(app, argc, argv);

    namespace fs = std::experimental::filesystem;
    fs::path graph_path(std::move(graph_raw_path));

    if (!fs::exists(graph_path)) {
        std::cerr << "Path to file: " << graph_path.c_str() << " does not exist!" << std::endl;
        return -1;
    }

    omp_set_num_threads(omp_thread_count);
    assert(omp_thread_count == omp_get_max_threads());
    gdsb::out("threads", omp_get_max_threads());

    dhb::Edges edges;
    std::mt19937 prng{42};
    std::uniform_real_distribution<float> distrib{0.0f, 1.0f};
    std::ifstream graph_input(graph_path);

    if (temporal_graph) {
        gdsb::TimestampedEdges<dhb::Edges> timestamped_edges;

        gdsb::read_temporal_graph<dhb::Vertex>(
            graph_input, weighted_graph_file, [&](unsigned int u, unsigned int v, unsigned int t) {
                auto w = distrib(prng);
                // Generate a random edge weight in [a, b).
                timestamped_edges.edges.push_back({u, {v, w, 0u}});
                timestamped_edges.timestamps.push_back(t);
            });

        timestamped_edges = gdsb::sort(timestamped_edges);
        edges = std::move(timestamped_edges.edges);
    } else {
        gdsb::read_graph_unweighted<dhb::Vertex>(graph_input,
                                                   [&](unsigned int u, unsigned int v) {
                                                       auto w = distrib(prng);
                                                       // Generate a random edge weight in [a, b).
                                                       edges.push_back({u, {v, w, 0u}});
                                                   });
    }

    gdsb::out("edges_read_in", edges.size());

    std::chrono::milliseconds duration;

    NetworKit::GraphBuilder graph_builder(dhb::graph::vertex_count(edges), true, true);

    unsigned long long memory_footprint_kb_before_construction = gdsb::memory_usage_in_kb();
    NetworKit::Graph graph = graph_builder.toGraph(false, false);
    unsigned long long memory_footprint_kb_after_construction = gdsb::memory_usage_in_kb();
    unsigned int vertex_count = graph.numberOfNodes();
    // unsigned int max_degree_before = ;
    // unsigned int max_degree_after = 0;
    // Vertex max_degree_node_before = 0;
    // Vertex max_degree_node_after = 0;
    unsigned int edge_count_before = graph.numberOfEdges();
    unsigned int edge_count_after = 0;

    auto collect_random_edges = [&] {
        std::vector<dhb::Edge> collected;
        collected.reserve(edges.size() / 2);

        std::uniform_int_distribution<size_t> edge_distrib(0, edges.size() - 1);
        for (size_t i = 0; i < edges.size() / 2; ++i) {
            auto edge = edges[edge_distrib(prng)];
            edge.target.data.weight = distrib(prng);
            collected.emplace_back(edge);
        }
        return collected;
    };

    if (insertion_routine == "insert") {
        auto add_all_edges = [&]() {
            for (dhb::Edge const& edge : edges) {
                graph.addEdge(edge.source, edge.target.vertex, edge.target.data.weight);
            }
            return true;
        };

        duration = gdsb::benchmark(add_all_edges);
    } else if (insertion_routine == "insert_bulk") {
        auto add_all_edges = [&]() {
            for (dhb::Edge const& edge : edges) {
                graph.addEdge(edge.source, edge.target.vertex, edge.target.data.weight);
            }
            return true;
        };

        duration = gdsb::benchmark(add_all_edges);
    } else if (insertion_routine == "temporal_insertions") {
        auto set_weight_for_all = [&]() {
            for (dhb::Edge const& edge : edges) {
                graph.setWeight(edge.source, edge.target.vertex, edge.target.data.weight);
            }
            return true;
        };

        duration = gdsb::benchmark(set_weight_for_all);
    } else if (insertion_routine == "temporal_updates") {
        auto set_weight_for_all = [&](dhb::Edges& collected) {
            for (const auto& edge : collected)
                graph.setWeight(edge.source, edge.target.vertex, edge.target.data.weight);
            return true;
        };

        // Add all edges before updates/deletions (time is not measured).
        for (auto& edge : edges)
            graph.addEdge(edge.source, edge.target.vertex, edge.target.data.weight);

        auto collected = collect_random_edges();
        duration = gdsb::benchmark(set_weight_for_all, collected);
    } else if (insertion_routine == "temporal_deletions") {
        auto remove_all = [&](dhb::Edges& collected) {
            for (const auto& edge : collected) {
                if (graph.hasEdge(edge.source, edge.target.vertex))
                    graph.removeEdge(edge.source, edge.target.vertex);
            }
            return true;
        };

        // Add all edges before updates/deletions (time is not measured).
        for (auto& edge : edges)
            graph.addEdge(edge.source, edge.target.vertex, edge.target.data.weight);

        auto collected = collect_random_edges();
        duration = gdsb::benchmark(remove_all, collected);
    } else if (insertion_routine == "bfs") {
		size_t num_iters = 100;
		size_t num_inserts = 100000;
		size_t off = 0;

		for (size_t i = 0; i < edges.size(); ++i) {
			if (off + num_iters * num_inserts >= edges.size())
				break;
			graph.setWeight(edges[off].source, edges[off].target.vertex, edges[off].target.data.weight);
			++off;
		}

		// Same approach as NetworKit::Traversal::BFSfrom.
		auto matrixBFSfrom = [&](dhb::Vertex r, auto f) {
			std::vector<bool> marked(vertex_count);
			std::queue<dhb::Vertex> q, qNext;
			uint64_t dist = 0;
			// enqueue start nodes
			q.push(r);
			marked[r] = true;
			do {
				const auto u = q.front();
				q.pop();
				// apply function
				f(u, dist);
				graph.forNeighborsOf(u, [&] (dhb::Vertex v) {
					if (!marked[v]) {
						qNext.push(v);
						marked[v] = true;
					}
				});
				if (q.empty() && !qNext.empty()) {
					q.swap(qNext);
					++dist;
				}
			} while (!q.empty());
		};

		duration = gdsb::benchmark([&] {
			uint64_t sum_of_dists = 0;
			std::uniform_int_distribution<dhb::Vertex> bfs_distrib(0, vertex_count - 1);
			for (size_t l = 0; l < num_iters; ++l) {
				for (size_t i = 0; i < num_inserts; ++i) {
					if (off >= edges.size())
						break;
					graph.setWeight(edges[off].source, edges[off].target.vertex, edges[off].target.data.weight);
					++off;
				}

				auto u = bfs_distrib(prng);
				matrixBFSfrom(u, [&](auto, auto dist) { sum_of_dists += dist; });
			}
			gdsb::out("sum_of_dists", sum_of_dists);
			return true;
		});
    } else if (insertion_routine == "spgemm") {
		std::vector<size_t> row_ptrs(vertex_count + 1);
		std::vector<dhb::Vertex> col_inds(edges.size());
		std::vector<double> vals(edges.size());
		std::sort(edges.begin(), edges.end(),
				  [](auto l, auto r) { return l.source < r.source; });
		size_t l = 0;
		row_ptrs.push_back(0);
		for (size_t i = 0; i < vertex_count; ++i) {
			while (edges[l].source == i) {
				col_inds[l] = edges[l].target.vertex;
				vals[l] = edges[l].target.data.weight;
				++l;
			}
			row_ptrs[i + 1] = l;
		}
		if (l != edges.size())
			throw std::runtime_error("corrupted CSR");

		auto for_row = [&](dhb::Vertex i, auto f) {
			for (size_t p = row_ptrs[i]; p != row_ptrs[i + 1]; ++p)
				f(col_inds[p], vals[p]);
		};

		duration = gdsb::benchmark([&] {
			size_t nnz = 0; // Stop after a number of non-zeros.
			for (dhb::Vertex i = 0; i < vertex_count; ++i) {
				if (nnz >= 100'000'000)
					break;
				for_row(i, [&](dhb::Vertex j, double l_val) {
					for_row(j, [&](dhb::Vertex k, double r_val) {
						auto prod = l_val * r_val;
						graph.increaseWeight(i, k, prod);
					});
				});
			}
			return true;
		});
    } else {
        std::cerr << "Insertion routine [" << insertion_routine << "] unknown!" << std::endl;
        return -7;
    }

    unsigned long long memory_footprint_kb_after_insertion = gdsb::memory_usage_in_kb();
    edge_count_after = graph.numberOfEdges();

    gdsb::out("experiment", experiment_name);
    gdsb::out("graph", graph_path.filename());
    gdsb::out("insertion_routine", insertion_routine);
    gdsb::out("format", "NetworKit");
    gdsb::out("vertex_count", graph.numberOfNodes());
    gdsb::out("edge_count_before", edge_count_before);
    gdsb::out("memory_footprint_kb_before_construction", memory_footprint_kb_before_construction);
    gdsb::out("memory_footprint_kb_after_construction", memory_footprint_kb_after_construction);
    gdsb::out("memory_footprint_kb_after_insertion", memory_footprint_kb_after_insertion);
    gdsb::out("edge_count_after", edge_count_after);
    gdsb::out("duration_ms", duration.count());

    return 0;
}
