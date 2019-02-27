#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <vector>
#include <algorithm>
#include <limits>

#include <iomanip>

namespace LightGBM {
/*!
* \brief Objective function for Lambdrank with NDCG
*/
class LambdarankNDCG: public ObjectiveFunction {
public:
  explicit LambdarankNDCG(const ObjectiveConfig& config) {
    sigmoid_ = static_cast<double>(config.sigmoid); /// 1.0
    // initialize DCG calculator
    DCGCalculator::Init(config.label_gain);
    // copy lable gain to local
    for (auto gain : config.label_gain) {
      label_gain_.push_back(static_cast<double>(gain));
    }
    label_gain_.shrink_to_fit();
    // will optimize NDCG@optimize_pos_at_
    optimize_pos_at_ = config.max_position;
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }

    /// get number of threads
    #pragma omp parallel
    #pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }

    std::cout << "num_threads_: " << num_threads_ << std::endl;
  }

  explicit LambdarankNDCG(const std::vector<std::string>&) {

  }

  ~LambdarankNDCG() {

  }
  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    DCGCalculator::CheckLabel(label_, num_data_);
    // get weights
    weights_ = metadata.weights();
    // get ranks
    ranks_ = metadata.ranks();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Lambdarank tasks require query information");
    }
    num_queries_ = metadata.num_queries();
    // cache inverse max DCG, avoid computation many times
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(optimize_pos_at_,
        label_ + query_boundaries_[i],
        query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct sigmoid table to speed up sigmoid transform
    ConstructSigmoidTable();
    // init position biases
    InitPositionBiases(); ///
    // init position gradients
    InitPositionGradients(); //
    std::cout << "" << std::endl;
    std::cout << std::setw(10) << "position" 
              << std::setw(15) << "bias_i"
              << std::setw(15) << "bias_j"
             
              << std::setw(15) << "i_cost"
              << std::setw(15) << "j_cost"
              << std::endl;
    for (size_t i = 0; i < _position_bins; ++i) { ///
      std::cout << std::setw(10) << i
                << std::setw(15) << i_biases_pow_[i]
                << std::setw(15) << j_biases_pow_[i]
                
                << std::setw(15) << i_costs_[i]
                << std::setw(15) << j_costs_[i]
                << std::endl;
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    #pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      GetGradientsForOneQuery(score, gradients, hessians, i);
    }

    UpdatePositionBiases(); // Finish one epoch, update the position bias
  }

  inline void GetGradientsForOneQuery(const double* score,
              score_t* lambdas, score_t* hessians, data_size_t query_id) const {
    const int tid = omp_get_thread_num(); // get thread ID

    // get doc boundary for current query
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt =
      query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // add pointers with offset
    const label_t* label = label_ + start;
    score += start;
    lambdas += start;
    hessians += start;
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx.emplace_back(i);
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
             [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double wrost_score = score[sorted_idx[worst_idx]];
    // start accmulate lambdas by pairs
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t high = sorted_idx[i];
      const int high_label = static_cast<int>(label[high]);
      const int high_rank = static_cast<int>(std::min(ranks_[start + high], _position_bins - 1)); /// high rank !!!-1
      // std::cout << "high_rank: " << high_rank << std::endl;
      const double high_score = score[high];
      if (high_score == kMinScore) { continue; }
      const double high_label_gain = label_gain_[high_label]; /// 2^high_label - 1, high_label=0,1,2,3,4,5
      const double high_discount = DCGCalculator::GetDiscount(i); /// 1 / log2(2 + i)
      double high_sum_lambda = 0.0;
      double high_sum_hessian = 0.0;
      double high_sum_cost_i = 0.0; ///
      int pair_num = 0; ///
      for (data_size_t j = 0; j < cnt; ++j) {
        // skip same data
        if (i == j) { continue; }
        const data_size_t low = sorted_idx[j];
        const int low_label = static_cast<int>(label[low]);
        const int low_rank = static_cast<int>(std::min(ranks_[start + low], _position_bins - 1)); /// high rank !!!-1
        const double low_score = score[low];
        // only consider pair with different label
        if (high_label <= low_label || low_score == kMinScore) { continue; } /// i is more relevant than j

        const double delta_score = high_score - low_score;

        const double low_label_gain = label_gain_[low_label]; /// 2^low_label - 1, low_label=0,1,2,3,4,5
        const double low_discount = DCGCalculator::GetDiscount(j); /// 1 / log2(2 + j)
        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain; /// 2^high_label - 2^low_label, high_label>low_label
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount); /// |1/log2(2+i) - 1/log2(2+j)|
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg; /// |deltaNDCG|

        // regular the delta_pair_NDCG by score distance
        if (high_label != low_label && best_score != wrost_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score); /// sigma / (1 + e^(sigma*(si-sj)))
        double p_hessian = p_lambda * (2.0f - p_lambda); /// sigma=2
        double p_cost = log(2.0f / (2.0f - p_lambda)) * delta_pair_NDCG; /// log(1+e^(-sigma*(si-sj)))
        // update
        p_lambda *= -delta_pair_NDCG / i_biases_pow_[high_rank] / j_biases_pow_[low_rank]; /// -|deltaNDCG|*sigma/(1 + e^(sigma*(si-sj)))/bias, 梯度而不是负梯度
        p_hessian *= 2 * delta_pair_NDCG / i_biases_pow_[high_rank] / j_biases_pow_[low_rank]; ///
        double p_cost_i = p_cost / j_biases_pow_[low_rank]; ///
        double p_cost_j = p_cost / i_biases_pow_[high_rank]; ///
	
	high_sum_cost_i += p_cost_i;
	j_costs_buffer_[tid][low_rank] += p_cost_j;
        position_cnts_buffer_[tid][high_rank] += 1LL; /// Only consider clicked pair to conduct normalization
        pair_num += 1; 
        high_sum_lambda += p_lambda; /// Add lambda to position i
        high_sum_hessian += p_hessian;
        lambdas[low] -= static_cast<score_t>(p_lambda);  /// Minus lambda for position j
        hessians[low] += static_cast<score_t>(p_hessian);
      }
      // update
      lambdas[high] += static_cast<score_t>(high_sum_lambda); /// accumulate lambda gradient
      hessians[high] += static_cast<score_t>(high_sum_hessian);
      i_costs_buffer_[tid][high_rank] += high_sum_cost_i; ///
      
    }

    // calculate position score, lambda
    for (data_size_t i = 0; i < cnt; ++i) { ///
      const int rank = static_cast<int>(std::min(ranks_[start + i], _position_bins - 1));
      position_scores_buffer_[tid][rank] += score[i];
      position_lambdas_buffer_[tid][rank] += lambdas[i];
    }
  }


  inline double GetSigmoid(double score) const { /// sigma/(1+e^(sigma*score)), sigma=2
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too big, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) * sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2; /// -50/1/2=-25
    max_sigmoid_input_ = -min_sigmoid_input_; /// 25
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
      _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_); /// 1024*1024/50
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_; /// [-25,25)
      sigmoid_table_[i] = 2.0f / (1.0f + std::exp(2.0f * score * sigmoid_)); /// sigma/(1+e^(sigma*s)), sigma=2
    }
  }

  void InitPositionBiases() { ///
    i_biases_.resize(_position_bins);
    i_biases_pow_.resize(_position_bins);
    j_biases_.resize(_position_bins);
    j_biases_pow_.resize(_position_bins);
    for (size_t i = 0; i < _position_bins; ++i) {
      i_biases_[i] = 1.0f;
      i_biases_pow_[i] = 1.0f;
      j_biases_[i] = 1.0f;
      j_biases_pow_[i] = 1.0f;
    }
  }

  void InitPositionGradients() { ///
    position_cnts_.resize(_position_bins);
    position_scores_.resize(_position_bins);
    position_lambdas_.resize(_position_bins);
    i_costs_.resize(_position_bins);
    j_costs_.resize(_position_bins);
    for (size_t i = 0; i < _position_bins; ++i) {
      position_cnts_[i] = 0LL;
      position_scores_[i] = 0.0f;
      position_lambdas_[i] = 0.0f;
      i_costs_[i] = 0.0f;
      j_costs_[i] = 0.0f;
    }

    for (int i = 0; i < num_threads_; i++) {
      position_cnts_buffer_.emplace_back(_position_bins, 0LL);
      position_scores_buffer_.emplace_back(_position_bins, 0.0f);
      position_lambdas_buffer_.emplace_back(_position_bins, 0.0f);
      i_costs_buffer_.emplace_back(_position_bins, 0.0f);
      j_costs_buffer_.emplace_back(_position_bins, 0.0f);
    }
  }

  void UpdatePositionBiases() const {
    // accumulate the parallel results
    for (int i = 0; i < num_threads_; i++) {
      for (size_t j = 0; j < _position_bins; ++j) {
        position_cnts_[j] += position_cnts_buffer_[i][j];
        position_scores_[j] += position_scores_buffer_[i][j];
        position_lambdas_[j] += position_lambdas_buffer_[i][j];
        i_costs_[j] += i_costs_buffer_[i][j];
        j_costs_[j] += j_costs_buffer_[i][j];
      }
    }

    long long position_cnts_sum = 0LL;
    for (size_t i = 0; i < _position_bins; ++i) {
      position_cnts_sum += position_cnts_[i];
    }
    std::cout << "" << std::endl;
    std::cout << "eta: " << _eta << ", pair_cnt_sum: " << position_cnts_sum << std::endl;
    std::cout << std::setw(10) << "position" 
              << std::setw(15) << "bias_i"
              << std::setw(15) << "bias_j"
              << std::setw(15) << "score" 
              << std::setw(15) << "lambda" 
              << std::setw(15) << "high_pair_cnt"
              << std::setw(15) << "i_cost"
              << std::setw(15) << "j_cost"
              << std::endl;
    for (size_t i = 0; i < _position_bins; ++i) { ///
      std::cout << std::setw(10) << i
                << std::setw(15) << i_biases_pow_[i]
                << std::setw(15) << j_biases_pow_[i]
                << std::setw(15) << position_scores_[i] / num_queries_
                << std::setw(15) << - position_lambdas_[i] / num_queries_
                << std::setw(15) << 1.0f * position_cnts_[i] / position_cnts_sum
                << std::setw(15) << i_costs_[i] / position_cnts_sum
                << std::setw(15) << j_costs_[i] / position_cnts_sum
                << std::endl;
    }

    // Update bias
    for (size_t i = 0; i < _position_bins; ++i) { /// 
      i_biases_[i] = i_costs_[i] / i_costs_[0];
      i_biases_pow_[i] = pow(i_biases_[i], _eta);
    }
    for (size_t i = 0; i < _position_bins; ++i) { /// 
      j_biases_[i] = j_costs_[i] / j_costs_[0];
      j_biases_pow_[i] = pow(j_biases_[i], _eta);
    }
    // Clear Buffer
    for (size_t i = 0; i < _position_bins; ++i) { ///
      position_cnts_[i] = 0LL;
      position_scores_[i] = 0.0f;
      position_lambdas_[i] = 0.0f;
      i_costs_[i] = 0.0f;
      j_costs_[i] = 0.0f;
    }

    for (int i = 0; i < num_threads_; i++) {
      for (size_t j = 0; j < _position_bins; ++j) {
        position_cnts_buffer_[i][j] = 0LL;
        position_scores_buffer_[i][j] = 0.0f;
        position_lambdas_buffer_[i][j] = 0.0f;
        i_costs_buffer_[i][j] = 0.0f;
        j_costs_buffer_[i][j] = 0.0f;
      }
    }
  }

  const char* GetName() const override {
    return "lambdarank";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

private:
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Simgoid param */
  double sigmoid_;
  /*! \brief Optimized NDCG@ */
  int optimize_pos_at_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Pointer of weights */
  const size_t* ranks_; ///
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;
  /*! \brief position biases */
  mutable std::vector<label_t> i_biases_; /// mutable
  /*! \brief pow position biases */
  mutable std::vector<label_t> i_biases_pow_; ///


  mutable std::vector<label_t> j_biases_; /// mutable
  /*! \brief pow position biases */
  mutable std::vector<label_t> j_biases_pow_; ///

  /*! \brief position cnts */
  mutable std::vector<long long> position_cnts_; ///
  mutable std::vector<std::vector<long long>> position_cnts_buffer_; ///
  /*! \brief position scores */
  mutable std::vector<label_t> position_scores_; ///
  mutable std::vector<std::vector<label_t>> position_scores_buffer_; ///
  /*! \brief position lambdas */
  mutable std::vector<label_t> position_lambdas_; ///
  mutable std::vector<std::vector<label_t>> position_lambdas_buffer_; ///
  // mutable double position cost; ///
  mutable std::vector<label_t> i_costs_; ///
  mutable std::vector<std::vector<label_t>> i_costs_buffer_; ///

  mutable std::vector<label_t> j_costs_; ///
  mutable std::vector<std::vector<label_t>> j_costs_buffer_; ///

  /*! \brief Number of exponent */
  double _eta = 1.0 / (1 + 0.5); ///
  /*! \brief Number of positions */
  size_t _position_bins = 12; ///
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in sigmoid table */
  double sigmoid_table_idx_factor_;
  /*! \brief Number of threads */
  int num_threads_; ///
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
