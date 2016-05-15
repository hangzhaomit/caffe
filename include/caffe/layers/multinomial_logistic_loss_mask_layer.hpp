#ifndef CAFFE_MULTINOMIAL_LOGISTIC_LOSS_MASK_LAYER_HPP_
#define CAFFE_MULTINOMIAL_LOGISTIC_LOSS_MASK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultinomialLogisticLossMaskLayer : public LossLayer<Dtype> {
 public:
  explicit MultinomialLogisticLossMaskLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultinomialLogisticLossMask"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool has_ignore_label_;
  int ignore_label_;
  int outer_num_, inner_num_;
};

}  // namespace caffe

#endif  // CAFFE_MULTINOMIAL_LOGISTIC_LOSS_MASK_LAYER_HPP_
