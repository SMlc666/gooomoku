#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int8_t RULE_FORBID_BLACK_OVERLINE = 1;
constexpr int8_t RULE_FORBID_BLACK_DOUBLE_THREE = 2;
constexpr int8_t RULE_FORBID_BLACK_DOUBLE_FOUR = 4;
constexpr int OBS_PLANES = 7;
constexpr int WORD_BITS = 32;

struct StepResult {
  struct GameState {
    std::vector<int8_t> board;
    std::vector<uint32_t> black_words;
    std::vector<uint32_t> white_words;
    int8_t to_play = 1;
    int32_t last_action = -1;
    int32_t num_moves = 0;
    bool terminated = false;
    int8_t winner = 0;
    int8_t rule_flags = 0;
    int8_t swap_source_flag = 0;
    int8_t swap_applied_flag = 0;
  } next_state;

  float reward = 0.0f;
  bool done = false;
};

using GameState = StepResult::GameState;

struct LeafEdge {
  int parent = -1;
  int action = -1;
  float reward = 0.0f;
  bool done = false;
};

struct LeafTask {
  int root_idx = -1;
  int leaf_idx = -1;
  std::vector<LeafEdge> path;
  GameState leaf_state;
  bool needs_eval = false;
};

struct Node {
  GameState state;
  bool expanded = false;
  int visit_count = 0;

  std::vector<int> child_index;
  std::vector<int> child_visits;
  std::vector<int> child_virtual;
  std::vector<float> child_value_sum;
  std::vector<float> child_prior;
  std::vector<float> child_reward;
  std::vector<uint8_t> child_done;
  std::vector<uint8_t> legal;
};

struct SearchConfig {
  int num_simulations = 64;
  int max_num_considered_actions = 24;
  float gumbel_scale = 1.0f;
  float c_puct = 1.5f;
  float c_lcb = 0.0f;
  float virtual_loss = 1.0f;
  int eval_batch_size = 256;
  int num_threads = 0;
  uint64_t seed = 0;
};

inline int num_words_for_actions(int num_actions) {
  return (num_actions + WORD_BITS - 1) / WORD_BITS;
}

inline bool bit_is_set(const std::vector<uint32_t>& words, int action) {
  const int word_idx = action / WORD_BITS;
  const uint32_t bit_idx = static_cast<uint32_t>(action % WORD_BITS);
  const uint32_t word = words[word_idx];
  return ((word >> bit_idx) & uint32_t(1)) != 0;
}

inline void set_bit(std::vector<uint32_t>* words, int action) {
  const int word_idx = action / WORD_BITS;
  const uint32_t bit_idx = static_cast<uint32_t>(action % WORD_BITS);
  (*words)[word_idx] |= (uint32_t(1) << bit_idx);
}

float sample_gumbel(std::mt19937_64* rng) {
  std::uniform_real_distribution<double> unif(1e-6, 1.0 - 1e-6);
  const double u = unif(*rng);
  return static_cast<float>(-std::log(-std::log(u)));
}

void run_parallel_for(int n, int num_threads, const std::function<void(int)>& fn) {
  if (n <= 0) {
    return;
  }
  int workers = num_threads;
  if (workers <= 0) {
    workers = static_cast<int>(std::thread::hardware_concurrency());
    if (workers <= 0) {
      workers = 1;
    }
  }
  workers = std::max(1, std::min(workers, n));
  if (workers == 1) {
    for (int i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  std::atomic<int> next_idx{0};
  std::exception_ptr eptr = nullptr;
  std::mutex eptr_mu;

  for (int t = 0; t < workers; ++t) {
    pool.emplace_back([&]() {
      while (true) {
        if (eptr != nullptr) {
          return;
        }
        const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= n) {
          return;
        }
        try {
          fn(idx);
        } catch (...) {
          std::lock_guard<std::mutex> lock(eptr_mu);
          if (eptr == nullptr) {
            eptr = std::current_exception();
          }
          return;
        }
      }
    });
  }
  for (auto& worker : pool) {
    worker.join();
  }
  if (eptr != nullptr) {
    std::rethrow_exception(eptr);
  }
}

class GomokuBackend {
 public:
  explicit GomokuBackend(int board_size)
      : board_size_(board_size),
        num_actions_(board_size * board_size),
        num_words_(num_words_for_actions(num_actions_)) {
    if (board_size_ < 3) {
      throw std::invalid_argument("board_size must be >= 3");
    }
    build_line_tables();
  }

  py::array_t<bool> batch_legal_mask(py::dict states_dict) const {
    const auto states = states_from_dict(states_dict);
    const int batch = static_cast<int>(states.size());

    py::array_t<bool> out({batch, num_actions_});
    auto out_m = out.mutable_unchecked<2>();

    for (int b = 0; b < batch; ++b) {
      const auto& st = states[b];
      if (st.terminated) {
        for (int a = 0; a < num_actions_; ++a) {
          out_m(b, a) = false;
        }
        continue;
      }
      for (int a = 0; a < num_actions_; ++a) {
        const bool occ = bit_is_set(st.black_words, a) || bit_is_set(st.white_words, a);
        out_m(b, a) = !occ;
      }
    }

    return out;
  }

  py::array_t<float> batch_encode(py::dict states_dict) const {
    const auto states = states_from_dict(states_dict);
    const int batch = static_cast<int>(states.size());

    py::array_t<float> out({batch, board_size_, board_size_, OBS_PLANES});
    auto out_m = out.mutable_unchecked<4>();

    for (int b = 0; b < batch; ++b) {
      encode_state(states[b], b, &out_m);
    }

    return out;
  }

  py::tuple batch_step(py::dict states_dict, py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) const {
    const auto states = states_from_dict(states_dict);
    const int batch = static_cast<int>(states.size());

    auto action_b = actions.unchecked<1>();
    if (action_b.shape(0) != batch) {
      throw std::invalid_argument("actions.shape[0] must equal batch size");
    }

    std::vector<GameState> next_states;
    next_states.reserve(static_cast<size_t>(batch));
    py::array_t<float> rewards({batch});
    py::array_t<bool> done({batch});
    auto rewards_m = rewards.mutable_unchecked<1>();
    auto done_m = done.mutable_unchecked<1>();

    for (int b = 0; b < batch; ++b) {
      const StepResult sr = step_state(states[b], action_b(b));
      next_states.push_back(sr.next_state);
      rewards_m(b) = sr.reward;
      done_m(b) = sr.done;
    }

    return py::make_tuple(states_to_dict(next_states), rewards, done);
  }

  py::dict search_gumbel(
      py::dict states_dict,
      py::array_t<float, py::array::c_style | py::array::forcecast> root_prior_logits,
      py::array_t<float, py::array::c_style | py::array::forcecast> root_values,
      py::function evaluator,
      int num_simulations,
      int max_num_considered_actions,
      float gumbel_scale,
      float c_puct,
      float c_lcb,
      float virtual_loss,
      int eval_batch_size,
      int num_threads,
      uint64_t seed) const {
    SearchConfig cfg;
    cfg.num_simulations = std::max(1, num_simulations);
    cfg.max_num_considered_actions = std::max(1, max_num_considered_actions);
    cfg.gumbel_scale = gumbel_scale;
    cfg.c_puct = c_puct;
    cfg.c_lcb = c_lcb;
    cfg.virtual_loss = std::max(0.0f, virtual_loss);
    cfg.eval_batch_size = std::max(1, eval_batch_size);
    cfg.num_threads = num_threads;
    cfg.seed = seed;

    auto states = states_from_dict(states_dict);
    const int batch = static_cast<int>(states.size());

    auto root_logits_m = root_prior_logits.unchecked<2>();
    auto root_values_m = root_values.unchecked<1>();
    if (root_logits_m.shape(0) != batch || root_logits_m.shape(1) != num_actions_) {
      throw std::invalid_argument("root_prior_logits must have shape [batch, num_actions]");
    }
    if (root_values_m.shape(0) != batch) {
      throw std::invalid_argument("root_values must have shape [batch]");
    }

    struct SearchTree {
      std::vector<Node> nodes;
      std::vector<uint8_t> root_considered;
      mutable std::mutex mu;
    };

    std::vector<SearchTree> trees(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b) {
      auto& tree = trees[static_cast<size_t>(b)];
      tree.nodes.reserve(static_cast<size_t>(cfg.num_simulations + 2));
      Node root;
      root.state = states[static_cast<size_t>(b)];
      init_node_storage(&root);
      root.expanded = true;
      fill_node_priors_from_logits(&root, &tree.root_considered, root_logits_row(root_logits_m, b), cfg, b);
      tree.nodes.push_back(std::move(root));
    }

    std::vector<int> remaining(static_cast<size_t>(batch), cfg.num_simulations);
    int rr = 0;
    int total_remaining = cfg.num_simulations * batch;

    while (total_remaining > 0) {
      const int wave = std::min(cfg.eval_batch_size, total_remaining);
      std::vector<int> wave_roots(static_cast<size_t>(wave), 0);
      for (int i = 0; i < wave; ++i) {
        while (remaining[static_cast<size_t>(rr)] <= 0) {
          rr = (rr + 1) % batch;
        }
        wave_roots[static_cast<size_t>(i)] = rr;
        --remaining[static_cast<size_t>(rr)];
        rr = (rr + 1) % batch;
      }

      std::vector<LeafTask> tasks(static_cast<size_t>(wave));
      run_parallel_for(wave, cfg.num_threads, [&](int i) {
        const int root_idx = wave_roots[static_cast<size_t>(i)];
        std::mt19937_64 rng(cfg.seed + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL +
                            static_cast<uint64_t>(root_idx) * 0xbf58476d1ce4e5b9ULL);
        tasks[static_cast<size_t>(i)] = select_leaf(&trees[static_cast<size_t>(root_idx)], root_idx, cfg, &rng);
      });

      std::vector<int> eval_slots;
      eval_slots.reserve(static_cast<size_t>(wave));
      for (int i = 0; i < wave; ++i) {
        if (tasks[static_cast<size_t>(i)].needs_eval) {
          eval_slots.push_back(i);
        }
      }

      if (!eval_slots.empty()) {
        const int eval_n = static_cast<int>(eval_slots.size());
        py::array_t<float> obs_arr({eval_n, board_size_, board_size_, OBS_PLANES});
        auto obs_m = obs_arr.mutable_unchecked<4>();

        for (int i = 0; i < eval_n; ++i) {
          const auto& task = tasks[static_cast<size_t>(eval_slots[static_cast<size_t>(i)])];
          encode_state(task.leaf_state, i, &obs_m);
        }

        py::tuple eval_out;
        {
          py::gil_scoped_acquire gil;
          py::object ret = evaluator(obs_arr);
          eval_out = ret.cast<py::tuple>();
        }
        if (eval_out.size() != 2) {
          throw std::runtime_error("evaluator must return (logits, value)");
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> eval_logits =
            eval_out[0].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::array_t<float, py::array::c_style | py::array::forcecast> eval_values =
            eval_out[1].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        auto eval_logits_m = eval_logits.unchecked<2>();
        auto eval_values_m = eval_values.unchecked<1>();
        if (eval_logits_m.shape(0) != eval_n || eval_logits_m.shape(1) != num_actions_) {
          throw std::invalid_argument("evaluator logits must have shape [N, num_actions]");
        }
        if (eval_values_m.shape(0) != eval_n) {
          throw std::invalid_argument("evaluator values must have shape [N]");
        }

        run_parallel_for(eval_n, cfg.num_threads, [&](int i) {
          const int slot = eval_slots[static_cast<size_t>(i)];
          const auto& task = tasks[static_cast<size_t>(slot)];
          auto* tree = &trees[static_cast<size_t>(task.root_idx)];
          expand_and_backup(tree, task, eval_logits_row(eval_logits_m, i), eval_values_m(i), cfg);
        });
      }

      total_remaining -= wave;
    }

    py::array_t<float> action_weights({batch, num_actions_});
    py::array_t<int32_t> selected_actions({batch});
    py::array_t<float> root_value_out({batch});
    auto weights_m = action_weights.mutable_unchecked<2>();
    auto selected_m = selected_actions.mutable_unchecked<1>();
    auto root_value_m = root_value_out.mutable_unchecked<1>();

    for (int b = 0; b < batch; ++b) {
      auto& tree = trees[static_cast<size_t>(b)];
      std::lock_guard<std::mutex> lock(tree.mu);
      const Node& root = tree.nodes[0];
      int visit_sum = 0;
      for (int a = 0; a < num_actions_; ++a) {
        if (root.legal[static_cast<size_t>(a)] != 0) {
          visit_sum += root.child_visits[static_cast<size_t>(a)];
        }
      }

      int best_action = -1;
      int best_visits = -1;
      for (int a = 0; a < num_actions_; ++a) {
        float w = 0.0f;
        if (root.legal[static_cast<size_t>(a)] != 0) {
          if (visit_sum > 0) {
            w = static_cast<float>(root.child_visits[static_cast<size_t>(a)]) / static_cast<float>(visit_sum);
          }
        }
        weights_m(b, a) = w;
        const int v = root.child_visits[static_cast<size_t>(a)];
        if (v > best_visits) {
          best_visits = v;
          best_action = a;
        }
      }

      if (visit_sum == 0) {
        int legal_count = 0;
        for (int a = 0; a < num_actions_; ++a) {
          if (root.legal[static_cast<size_t>(a)] != 0) {
            ++legal_count;
          }
        }
        if (legal_count > 0) {
          const float u = 1.0f / static_cast<float>(legal_count);
          for (int a = 0; a < num_actions_; ++a) {
            weights_m(b, a) = root.legal[static_cast<size_t>(a)] != 0 ? u : 0.0f;
          }
          for (int a = 0; a < num_actions_; ++a) {
            if (root.legal[static_cast<size_t>(a)] != 0) {
              best_action = a;
              break;
            }
          }
        }
      }

      selected_m(b) = static_cast<int32_t>(best_action);
      root_value_m(b) = root_values_m(b);
    }

    py::dict out;
    out["action_weights"] = action_weights;
    out["selected_actions"] = selected_actions;
    out["root_values"] = root_value_out;
    return out;
  }

 private:
  int board_size_;
  int num_actions_;
  int num_words_;

  std::vector<std::array<int, 5>> lines5_;
  std::vector<std::array<int, 6>> lines6_;

  void build_line_tables() {
    lines5_.clear();
    lines6_.clear();

    auto add_lines = [&](int len, std::vector<std::array<int, 6>>* lines6_out,
                         std::vector<std::array<int, 5>>* lines5_out) {
      const std::array<std::pair<int, int>, 4> dirs = {{{0, 1}, {1, 0}, {1, 1}, {1, -1}}};
      for (const auto& dir : dirs) {
        const int dr = dir.first;
        const int dc = dir.second;
        for (int r = 0; r < board_size_; ++r) {
          for (int c = 0; c < board_size_; ++c) {
            const int er = r + (len - 1) * dr;
            const int ec = c + (len - 1) * dc;
            if (er < 0 || er >= board_size_ || ec < 0 || ec >= board_size_) {
              continue;
            }
            if (len == 5) {
              std::array<int, 5> w{};
              for (int i = 0; i < 5; ++i) {
                w[static_cast<size_t>(i)] = (r + i * dr) * board_size_ + (c + i * dc);
              }
              lines5_out->push_back(w);
            } else {
              std::array<int, 6> w{};
              for (int i = 0; i < 6; ++i) {
                w[static_cast<size_t>(i)] = (r + i * dr) * board_size_ + (c + i * dc);
              }
              lines6_out->push_back(w);
            }
          }
        }
      }
    };

    add_lines(5, &lines6_, &lines5_);
    add_lines(6, &lines6_, &lines5_);
  }

  static float safe_logit(float x) {
    if (!std::isfinite(x)) {
      return -1e30f;
    }
    return x;
  }

  void init_node_storage(Node* node) const {
    node->child_index.assign(static_cast<size_t>(num_actions_), -1);
    node->child_visits.assign(static_cast<size_t>(num_actions_), 0);
    node->child_virtual.assign(static_cast<size_t>(num_actions_), 0);
    node->child_value_sum.assign(static_cast<size_t>(num_actions_), 0.0f);
    node->child_prior.assign(static_cast<size_t>(num_actions_), 0.0f);
    node->child_reward.assign(static_cast<size_t>(num_actions_), 0.0f);
    node->child_done.assign(static_cast<size_t>(num_actions_), uint8_t(0));
    node->legal.assign(static_cast<size_t>(num_actions_), uint8_t(0));
  }

  bool has_five_from_words(const std::vector<uint32_t>& words) const {
    for (const auto& w : lines5_) {
      bool ok = true;
      for (int i = 0; i < 5; ++i) {
        if (!bit_is_set(words, w[static_cast<size_t>(i)])) {
          ok = false;
          break;
        }
      }
      if (ok) {
        return true;
      }
    }
    return false;
  }

  std::array<int8_t, 11> line_values_around_move(const std::vector<int8_t>& board, int row, int col, int dr, int dc) const {
    std::array<int8_t, 11> line{};
    for (int i = -5; i <= 5; ++i) {
      const int rr = row + i * dr;
      const int cc = col + i * dc;
      if (rr < 0 || rr >= board_size_ || cc < 0 || cc >= board_size_) {
        line[static_cast<size_t>(i + 5)] = int8_t(-1);
      } else {
        line[static_cast<size_t>(i + 5)] = board[static_cast<size_t>(rr * board_size_ + cc)];
      }
    }
    return line;
  }

  static bool direction_has_overline(const std::array<int8_t, 11>& line) {
    for (int s = 0; s <= 5; ++s) {
      bool all_black = true;
      for (int k = 0; k < 6; ++k) {
        if (line[static_cast<size_t>(s + k)] != int8_t(1)) {
          all_black = false;
          break;
        }
      }
      if (all_black) {
        return true;
      }
    }
    return false;
  }

  static bool direction_has_four(const std::array<int8_t, 11>& line) {
    for (int s = 1; s <= 5; ++s) {
      int black_n = 0;
      int empty_n = 0;
      int blocked_n = 0;
      for (int k = 0; k < 5; ++k) {
        const int8_t v = line[static_cast<size_t>(s + k)];
        if (v == int8_t(1)) {
          ++black_n;
        } else if (v == int8_t(0)) {
          ++empty_n;
        } else if (v == int8_t(-1)) {
          ++blocked_n;
        }
      }
      const int8_t left = line[static_cast<size_t>(s - 1)];
      const int8_t right = line[static_cast<size_t>(s + 5)];
      const bool exact_five_if_fill = (left != int8_t(1)) && (right != int8_t(1));
      if (black_n == 4 && empty_n == 1 && blocked_n == 0 && exact_five_if_fill) {
        return true;
      }
    }
    return false;
  }

  static bool direction_has_open_three(const std::array<int8_t, 11>& line) {
    for (int s = 1; s <= 4; ++s) {
      const int8_t left = line[static_cast<size_t>(s + 0)];
      const int8_t right = line[static_cast<size_t>(s + 5)];
      if (!(left == int8_t(0) && right == int8_t(0))) {
        continue;
      }
      int black_n = 0;
      int empty_n = 0;
      int blocked_n = 0;
      for (int k = 1; k <= 4; ++k) {
        const int8_t v = line[static_cast<size_t>(s + k)];
        if (v == int8_t(1)) {
          ++black_n;
        } else if (v == int8_t(0)) {
          ++empty_n;
        } else if (v == int8_t(-1)) {
          ++blocked_n;
        }
      }
      if (black_n == 3 && empty_n == 1 && blocked_n == 0) {
        return true;
      }
    }
    return false;
  }

  bool is_black_renju_forbidden(const std::vector<int8_t>& board_after, int row, int col, int8_t rule_flags) const {
    if (rule_flags == 0) {
      return false;
    }

    const std::array<std::pair<int, int>, 4> dirs = {{{0, 1}, {1, 0}, {1, 1}, {1, -1}}};
    bool overline = false;
    int four_dirs = 0;
    int open_three_dirs = 0;

    for (const auto& d : dirs) {
      const auto line = line_values_around_move(board_after, row, col, d.first, d.second);
      overline = overline || direction_has_overline(line);
      if (direction_has_four(line)) {
        ++four_dirs;
      }
      if (direction_has_open_three(line)) {
        ++open_three_dirs;
      }
    }

    const bool forbid_overline = ((rule_flags & RULE_FORBID_BLACK_OVERLINE) != 0) && overline;
    const bool forbid_double_four = ((rule_flags & RULE_FORBID_BLACK_DOUBLE_FOUR) != 0) && (four_dirs >= 2);
    const bool forbid_double_three = ((rule_flags & RULE_FORBID_BLACK_DOUBLE_THREE) != 0) && (open_three_dirs >= 2);
    return forbid_overline || forbid_double_four || forbid_double_three;
  }

  StepResult step_state(const GameState& state, int32_t action_in) const {
    StepResult out;
    out.next_state = state;

    const int action = static_cast<int>(action_in);
    const bool in_range = action >= 0 && action < num_actions_;
    const int safe_action = std::max(0, std::min(num_actions_ - 1, action));
    const bool legal_at_action = in_range &&
        (!bit_is_set(state.black_words, safe_action)) &&
        (!bit_is_set(state.white_words, safe_action));

    const int row = safe_action / board_size_;
    const int col = safe_action % board_size_;
    const bool can_play_proposed = (!state.terminated) && legal_at_action;

    std::vector<int8_t> board_after_proposed = state.board;
    if (can_play_proposed) {
      board_after_proposed[static_cast<size_t>(safe_action)] = state.to_play;
    }

    const bool forbidden_black_move =
        can_play_proposed && (state.to_play == int8_t(1)) &&
        is_black_renju_forbidden(board_after_proposed, row, col, state.rule_flags);

    const bool can_play = can_play_proposed && (!forbidden_black_move);
    const bool illegal_move = (!state.terminated) && ((!legal_at_action) || forbidden_black_move);

    if (can_play) {
      out.next_state.board = board_after_proposed;
      if (state.to_play == int8_t(1)) {
        set_bit(&out.next_state.black_words, safe_action);
      } else {
        set_bit(&out.next_state.white_words, safe_action);
      }
    }

    const std::vector<uint32_t>& player_words_after =
        (state.to_play == int8_t(1)) ? out.next_state.black_words : out.next_state.white_words;
    const bool win = can_play ? has_five_from_words(player_words_after) : false;

    out.next_state.num_moves = static_cast<int32_t>(state.num_moves + (can_play ? 1 : 0));
    const bool draw = (!win) && (out.next_state.num_moves >= num_actions_);
    out.next_state.terminated = state.terminated || win || draw || illegal_move;

    if (win) {
      out.next_state.winner = state.to_play;
    } else if (illegal_move) {
      out.next_state.winner = static_cast<int8_t>(-state.to_play);
    } else {
      out.next_state.winner = state.winner;
    }

    out.reward = win ? 1.0f : (illegal_move ? -1.0f : 0.0f);

    out.next_state.to_play = (can_play && (!out.next_state.terminated))
        ? static_cast<int8_t>(-state.to_play)
        : state.to_play;
    out.next_state.last_action = can_play ? static_cast<int32_t>(safe_action) : state.last_action;
    out.done = out.next_state.terminated;
    return out;
  }

  void encode_state(const GameState& st, int batch_idx, py::detail::unchecked_mutable_reference<float, 4>* out) const {
    const bool to_play_black = st.to_play == int8_t(1);

    for (int action = 0; action < num_actions_; ++action) {
      const bool black = bit_is_set(st.black_words, action);
      const bool white = bit_is_set(st.white_words, action);
      const bool mine = to_play_black ? black : white;
      const bool opp = to_play_black ? white : black;

      const int r = action / board_size_;
      const int c = action % board_size_;
      (*out)(batch_idx, r, c, 0) = mine ? 1.0f : 0.0f;
      (*out)(batch_idx, r, c, 1) = opp ? 1.0f : 0.0f;
      (*out)(batch_idx, r, c, 2) = 0.0f;
      (*out)(batch_idx, r, c, 3) = to_play_black ? 1.0f : 0.0f;
      (*out)(batch_idx, r, c, 4) = st.rule_flags != 0 ? 1.0f : 0.0f;
      (*out)(batch_idx, r, c, 5) = st.swap_source_flag != 0 ? 1.0f : 0.0f;
      (*out)(batch_idx, r, c, 6) = st.swap_applied_flag != 0 ? 1.0f : 0.0f;
    }

    if (st.last_action >= 0 && st.last_action < num_actions_) {
      const int lr = st.last_action / board_size_;
      const int lc = st.last_action % board_size_;
      (*out)(batch_idx, lr, lc, 2) = 1.0f;
    }
  }

  static void softmax_legal(
      const std::vector<float>& logits,
      const std::vector<uint8_t>& legal,
      std::vector<float>* priors_out) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t a = 0; a < logits.size(); ++a) {
      if (legal[a] == 0) {
        continue;
      }
      max_logit = std::max(max_logit, logits[a]);
    }

    float sum = 0.0f;
    for (size_t a = 0; a < logits.size(); ++a) {
      if (legal[a] == 0) {
        (*priors_out)[a] = 0.0f;
        continue;
      }
      const float p = std::exp(logits[a] - max_logit);
      (*priors_out)[a] = p;
      sum += p;
    }

    if (!(sum > 0.0f) || !std::isfinite(sum)) {
      int legal_count = 0;
      for (uint8_t v : legal) {
        if (v != 0) {
          ++legal_count;
        }
      }
      if (legal_count <= 0) {
        std::fill(priors_out->begin(), priors_out->end(), 0.0f);
      } else {
        const float u = 1.0f / static_cast<float>(legal_count);
        for (size_t a = 0; a < logits.size(); ++a) {
          (*priors_out)[a] = legal[a] != 0 ? u : 0.0f;
        }
      }
      return;
    }

    const float inv = 1.0f / sum;
    for (float& p : *priors_out) {
      p *= inv;
    }
  }

  std::vector<float> root_logits_row(const py::detail::unchecked_reference<float, 2>& root_logits_m, int row) const {
    std::vector<float> row_logits(static_cast<size_t>(num_actions_), -1e30f);
    for (int a = 0; a < num_actions_; ++a) {
      row_logits[static_cast<size_t>(a)] = safe_logit(root_logits_m(row, a));
    }
    return row_logits;
  }

  std::vector<float> eval_logits_row(const py::detail::unchecked_reference<float, 2>& logits_m, int row) const {
    std::vector<float> row_logits(static_cast<size_t>(num_actions_), -1e30f);
    for (int a = 0; a < num_actions_; ++a) {
      row_logits[static_cast<size_t>(a)] = safe_logit(logits_m(row, a));
    }
    return row_logits;
  }

  void fill_node_priors_from_logits(
      Node* node,
      std::vector<uint8_t>* root_considered,
      const std::vector<float>& logits,
      const SearchConfig& cfg,
      int root_idx) const {
    if (node->state.terminated) {
      std::fill(node->legal.begin(), node->legal.end(), uint8_t(0));
      std::fill(node->child_prior.begin(), node->child_prior.end(), 0.0f);
      if (root_considered != nullptr) {
        root_considered->assign(static_cast<size_t>(num_actions_), uint8_t(0));
      }
      node->expanded = true;
      return;
    }

    int legal_count = 0;
    for (int a = 0; a < num_actions_; ++a) {
      const bool occ = bit_is_set(node->state.black_words, a) || bit_is_set(node->state.white_words, a);
      const bool legal = !occ;
      node->legal[static_cast<size_t>(a)] = legal ? uint8_t(1) : uint8_t(0);
      legal_count += legal ? 1 : 0;
    }

    softmax_legal(logits, node->legal, &node->child_prior);

    if (root_considered != nullptr) {
      root_considered->assign(static_cast<size_t>(num_actions_), uint8_t(0));
      if (legal_count <= 0) {
        return;
      }

      const int k = std::max(1, std::min(cfg.max_num_considered_actions, legal_count));
      if (k >= legal_count) {
        for (int a = 0; a < num_actions_; ++a) {
          if (node->legal[static_cast<size_t>(a)] != 0) {
            (*root_considered)[static_cast<size_t>(a)] = uint8_t(1);
          }
        }
        return;
      }

      std::mt19937_64 rng(cfg.seed + static_cast<uint64_t>(root_idx) * 0x9e3779b97f4a7c15ULL + 0x243f6a8885a308d3ULL);
      std::vector<std::pair<float, int>> scored;
      scored.reserve(static_cast<size_t>(legal_count));
      for (int a = 0; a < num_actions_; ++a) {
        if (node->legal[static_cast<size_t>(a)] == 0) {
          continue;
        }
        const float g = sample_gumbel(&rng);
        const float s = logits[static_cast<size_t>(a)] + cfg.gumbel_scale * g;
        scored.push_back({s, a});
      }
      std::nth_element(
          scored.begin(),
          scored.begin() + k,
          scored.end(),
          [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
      std::sort(
          scored.begin(),
          scored.begin() + k,
          [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
      for (int i = 0; i < k; ++i) {
        (*root_considered)[static_cast<size_t>(scored[static_cast<size_t>(i)].second)] = uint8_t(1);
      }
    }

    node->expanded = true;
  }

  int select_action(
      const Node& node,
      const std::vector<uint8_t>* root_considered,
      int depth,
      const SearchConfig& cfg) const {
    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;

    for (int a = 0; a < num_actions_; ++a) {
      if (node.legal[static_cast<size_t>(a)] == 0) {
        continue;
      }
      if (depth == 0 && root_considered != nullptr && !root_considered->empty() &&
          (*root_considered)[static_cast<size_t>(a)] == 0) {
        continue;
      }

      const int n = node.child_visits[static_cast<size_t>(a)];
      const int vloss_n = node.child_virtual[static_cast<size_t>(a)];
      float q = 0.0f;
      if (n > 0) {
        q = node.child_value_sum[static_cast<size_t>(a)] / static_cast<float>(n);
      }
      if (cfg.c_lcb > 0.0f) {
        const float lcb = cfg.c_lcb * std::sqrt(std::max(1.0f, static_cast<float>(node.visit_count))) /
                          static_cast<float>(1 + n);
        q -= lcb;
      }
      if (cfg.virtual_loss > 0.0f && vloss_n > 0) {
        q -= cfg.virtual_loss * static_cast<float>(vloss_n);
      }

      const float u = cfg.c_puct * node.child_prior[static_cast<size_t>(a)] *
                      std::sqrt(static_cast<float>(node.visit_count + 1)) /
                      static_cast<float>(1 + n + vloss_n);
      const float score = q + u;
      if (score > best_score) {
        best_score = score;
        best_action = a;
      }
    }

    return best_action;
  }

  template <typename SearchTree>
  void backup_locked(SearchTree* tree, const std::vector<LeafEdge>& path, int leaf_idx, float leaf_value) const {
    if (leaf_idx >= 0 && leaf_idx < static_cast<int>(tree->nodes.size())) {
      tree->nodes[static_cast<size_t>(leaf_idx)].visit_count += 1;
    }

    float v = leaf_value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      Node& parent = tree->nodes[static_cast<size_t>(it->parent)];
      const int a = it->action;
      if (parent.child_virtual[static_cast<size_t>(a)] > 0) {
        parent.child_virtual[static_cast<size_t>(a)] -= 1;
      }
      const float q = it->reward + (it->done ? 0.0f : -v);
      parent.child_value_sum[static_cast<size_t>(a)] += q;
      parent.child_visits[static_cast<size_t>(a)] += 1;
      parent.visit_count += 1;
      v = q;
    }
  }

  template <typename SearchTree>
  LeafTask select_leaf(SearchTree* tree, int root_idx, const SearchConfig& cfg, std::mt19937_64* /*rng*/) const {
    std::lock_guard<std::mutex> lock(tree->mu);

    LeafTask task;
    task.root_idx = root_idx;

    int cur = 0;
    int depth = 0;
    while (true) {
      Node& node = tree->nodes[static_cast<size_t>(cur)];
      if (node.state.terminated) {
        backup_locked(tree, task.path, cur, 0.0f);
        task.needs_eval = false;
        task.leaf_idx = cur;
        return task;
      }

      if (!node.expanded) {
        task.leaf_idx = cur;
        task.leaf_state = node.state;
        task.needs_eval = true;
        return task;
      }

      const int action = select_action(
          node,
          depth == 0 ? &tree->root_considered : nullptr,
          depth,
          cfg);
      if (action < 0) {
        node.state.terminated = true;
        node.state.winner = 0;
        backup_locked(tree, task.path, cur, 0.0f);
        task.needs_eval = false;
        task.leaf_idx = cur;
        return task;
      }

      node.child_virtual[static_cast<size_t>(action)] += 1;

      int child_idx = node.child_index[static_cast<size_t>(action)];
      float edge_reward = node.child_reward[static_cast<size_t>(action)];
      bool edge_done = node.child_done[static_cast<size_t>(action)] != 0;
      if (child_idx < 0) {
        const StepResult sr = step_state(node.state, action);

        Node child;
        child.state = sr.next_state;
        init_node_storage(&child);
        child.expanded = child.state.terminated;
        if (child.expanded) {
          std::fill(child.legal.begin(), child.legal.end(), uint8_t(0));
          std::fill(child.child_prior.begin(), child.child_prior.end(), 0.0f);
        }

        child_idx = static_cast<int>(tree->nodes.size());
        tree->nodes.push_back(std::move(child));

        node.child_index[static_cast<size_t>(action)] = child_idx;
        node.child_reward[static_cast<size_t>(action)] = sr.reward;
        node.child_done[static_cast<size_t>(action)] = sr.done ? uint8_t(1) : uint8_t(0);

        edge_reward = sr.reward;
        edge_done = sr.done;
      }

      task.path.push_back(LeafEdge{cur, action, edge_reward, edge_done});
      cur = child_idx;
      depth += 1;

      if (edge_done) {
        backup_locked(tree, task.path, cur, 0.0f);
        task.needs_eval = false;
        task.leaf_idx = cur;
        return task;
      }
    }
  }

  template <typename SearchTree>
  void expand_and_backup(
      SearchTree* tree,
      const LeafTask& task,
      const std::vector<float>& logits,
      float value,
      const SearchConfig& cfg) const {
    std::lock_guard<std::mutex> lock(tree->mu);
    if (task.leaf_idx < 0 || task.leaf_idx >= static_cast<int>(tree->nodes.size())) {
      throw std::runtime_error("invalid leaf index during expand_and_backup");
    }

    Node& leaf = tree->nodes[static_cast<size_t>(task.leaf_idx)];
    if (!leaf.expanded) {
      fill_node_priors_from_logits(&leaf, nullptr, logits, cfg, task.root_idx);
    }
    backup_locked(tree, task.path, task.leaf_idx, value);
  }

  std::vector<GameState> states_from_dict(py::dict states_dict) const {
    auto board = states_dict[py::str("board")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    auto black_words = states_dict[py::str("black_words")].cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
    auto white_words = states_dict[py::str("white_words")].cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
    auto to_play = states_dict[py::str("to_play")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    auto last_action = states_dict[py::str("last_action")].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto num_moves = states_dict[py::str("num_moves")].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto terminated = states_dict[py::str("terminated")].cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>();
    auto winner = states_dict[py::str("winner")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    auto rule_flags = states_dict[py::str("rule_flags")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    auto swap_source_flag = states_dict[py::str("swap_source_flag")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    auto swap_applied_flag = states_dict[py::str("swap_applied_flag")].cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();

    auto board_m = board.unchecked<3>();
    auto black_words_m = black_words.unchecked<2>();
    auto white_words_m = white_words.unchecked<2>();
    auto to_play_m = to_play.unchecked<1>();
    auto last_action_m = last_action.unchecked<1>();
    auto num_moves_m = num_moves.unchecked<1>();
    auto terminated_m = terminated.unchecked<1>();
    auto winner_m = winner.unchecked<1>();
    auto rule_flags_m = rule_flags.unchecked<1>();
    auto swap_source_m = swap_source_flag.unchecked<1>();
    auto swap_applied_m = swap_applied_flag.unchecked<1>();

    const int batch = static_cast<int>(board_m.shape(0));
    if (board_m.shape(1) != board_size_ || board_m.shape(2) != board_size_) {
      throw std::invalid_argument("board must have shape [batch, board_size, board_size]");
    }
    if (black_words_m.shape(0) != batch || black_words_m.shape(1) != num_words_) {
      throw std::invalid_argument("black_words shape mismatch");
    }
    if (white_words_m.shape(0) != batch || white_words_m.shape(1) != num_words_) {
      throw std::invalid_argument("white_words shape mismatch");
    }

    const auto check_batch_1d = [&](int s, const std::string& name) {
      if (s != batch) {
        throw std::invalid_argument(name + " must have length=batch");
      }
    };
    check_batch_1d(static_cast<int>(to_play_m.shape(0)), "to_play");
    check_batch_1d(static_cast<int>(last_action_m.shape(0)), "last_action");
    check_batch_1d(static_cast<int>(num_moves_m.shape(0)), "num_moves");
    check_batch_1d(static_cast<int>(terminated_m.shape(0)), "terminated");
    check_batch_1d(static_cast<int>(winner_m.shape(0)), "winner");
    check_batch_1d(static_cast<int>(rule_flags_m.shape(0)), "rule_flags");
    check_batch_1d(static_cast<int>(swap_source_m.shape(0)), "swap_source_flag");
    check_batch_1d(static_cast<int>(swap_applied_m.shape(0)), "swap_applied_flag");

    std::vector<GameState> states;
    states.reserve(static_cast<size_t>(batch));

    for (int b = 0; b < batch; ++b) {
      GameState st;
      st.board.resize(static_cast<size_t>(num_actions_));
      st.black_words.resize(static_cast<size_t>(num_words_));
      st.white_words.resize(static_cast<size_t>(num_words_));

      for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
          const int idx = r * board_size_ + c;
          st.board[static_cast<size_t>(idx)] = board_m(b, r, c);
        }
      }
      for (int w = 0; w < num_words_; ++w) {
        st.black_words[static_cast<size_t>(w)] = black_words_m(b, w);
        st.white_words[static_cast<size_t>(w)] = white_words_m(b, w);
      }

      st.to_play = to_play_m(b);
      st.last_action = last_action_m(b);
      st.num_moves = num_moves_m(b);
      st.terminated = terminated_m(b);
      st.winner = winner_m(b);
      st.rule_flags = rule_flags_m(b);
      st.swap_source_flag = swap_source_m(b);
      st.swap_applied_flag = swap_applied_m(b);
      states.push_back(std::move(st));
    }

    return states;
  }

  py::dict states_to_dict(const std::vector<GameState>& states) const {
    const int batch = static_cast<int>(states.size());

    py::array_t<int8_t> board({batch, board_size_, board_size_});
    py::array_t<uint32_t> black_words({batch, num_words_});
    py::array_t<uint32_t> white_words({batch, num_words_});
    py::array_t<int8_t> to_play({batch});
    py::array_t<int32_t> last_action({batch});
    py::array_t<int32_t> num_moves({batch});
    py::array_t<bool> terminated({batch});
    py::array_t<int8_t> winner({batch});
    py::array_t<int8_t> rule_flags({batch});
    py::array_t<int8_t> swap_source_flag({batch});
    py::array_t<int8_t> swap_applied_flag({batch});

    auto board_m = board.mutable_unchecked<3>();
    auto black_words_m = black_words.mutable_unchecked<2>();
    auto white_words_m = white_words.mutable_unchecked<2>();
    auto to_play_m = to_play.mutable_unchecked<1>();
    auto last_action_m = last_action.mutable_unchecked<1>();
    auto num_moves_m = num_moves.mutable_unchecked<1>();
    auto terminated_m = terminated.mutable_unchecked<1>();
    auto winner_m = winner.mutable_unchecked<1>();
    auto rule_flags_m = rule_flags.mutable_unchecked<1>();
    auto swap_source_m = swap_source_flag.mutable_unchecked<1>();
    auto swap_applied_m = swap_applied_flag.mutable_unchecked<1>();

    for (int b = 0; b < batch; ++b) {
      const auto& st = states[static_cast<size_t>(b)];
      for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
          board_m(b, r, c) = st.board[static_cast<size_t>(r * board_size_ + c)];
        }
      }
      for (int w = 0; w < num_words_; ++w) {
        black_words_m(b, w) = st.black_words[static_cast<size_t>(w)];
        white_words_m(b, w) = st.white_words[static_cast<size_t>(w)];
      }

      to_play_m(b) = st.to_play;
      last_action_m(b) = st.last_action;
      num_moves_m(b) = st.num_moves;
      terminated_m(b) = st.terminated;
      winner_m(b) = st.winner;
      rule_flags_m(b) = st.rule_flags;
      swap_source_m(b) = st.swap_source_flag;
      swap_applied_m(b) = st.swap_applied_flag;
    }

    py::dict out;
    out["board"] = board;
    out["black_words"] = black_words;
    out["white_words"] = white_words;
    out["to_play"] = to_play;
    out["last_action"] = last_action;
    out["num_moves"] = num_moves;
    out["terminated"] = terminated;
    out["winner"] = winner;
    out["rule_flags"] = rule_flags;
    out["swap_source_flag"] = swap_source_flag;
    out["swap_applied_flag"] = swap_applied_flag;
    return out;
  }
};

}  // namespace

PYBIND11_MODULE(gooomoku_cpp, m) {
  m.doc() = "High-performance C++ Gomoku backend with batched Gumbel MCTS and virtual loss.";

  py::class_<GomokuBackend>(m, "GomokuBackend")
      .def(py::init<int>(), py::arg("board_size"))
      .def("batch_legal_mask", &GomokuBackend::batch_legal_mask, py::arg("states"))
      .def("batch_encode", &GomokuBackend::batch_encode, py::arg("states"))
      .def("batch_step", &GomokuBackend::batch_step, py::arg("states"), py::arg("actions"))
      .def(
          "search_gumbel",
          &GomokuBackend::search_gumbel,
          py::arg("states"),
          py::arg("root_prior_logits"),
          py::arg("root_values"),
          py::arg("evaluator"),
          py::arg("num_simulations"),
          py::arg("max_num_considered_actions"),
          py::arg("gumbel_scale") = 1.0f,
          py::arg("c_puct") = 1.5f,
          py::arg("c_lcb") = 0.0f,
          py::arg("virtual_loss") = 1.0f,
          py::arg("eval_batch_size") = 256,
          py::arg("num_threads") = 0,
          py::arg("seed") = 0ULL);
}
