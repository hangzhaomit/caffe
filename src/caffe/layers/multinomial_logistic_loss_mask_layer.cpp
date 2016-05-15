#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multinomial_logistic_loss_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultinomialLogisticLossMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
    DCHECK_GT(ignore_label_, 0) << "Ignore label index should be larger than 0";
  }
}

template <typename Dtype>
void MultinomialLogisticLossMaskLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), bottom[0]->height());
  CHECK_EQ(bottom[1]->width(), bottom[0]->width());

  // check shape of weight mask
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), bottom[0]->height());
  CHECK_EQ(bottom[2]->width(), bottom[0]->width());

  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->count(2);
}

template <typename Dtype>
void MultinomialLogisticLossMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_mask = bottom[2]->cpu_data();
  int dim = bottom[0]->count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, bottom[0].shape(1));

      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      if (label_value != 0){
        Dtype prob = std::max(
          bottom_data[i * dim + label_value * inner_num_ + j]
          * bottom_mask[i * inner_num_+ j]
          , Dtype(kLOG_THRESHOLD));  
      }
      else{
        Dtype prob = std::max(
            bottom_data[i * dim + label_value * inner_num_ + j]
            * (1 - bottom_mask[i * inner_num_+ j])
            , Dtype(kLOG_THRESHOLD));    
      }
      
      loss -= log(prob);
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / (std::max(Dtype(1.0), Dtype(count)));
}

template <typename Dtype>
void MultinomialLogisticLossMaskLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* mask_diff = bottom[2]->mutable_cpu_diff();

  int dim = bottom[0]->count() / outer_num_;
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  caffe_set(bottom[2]->count(), Dtype(0), mask_diff);
    
  if (propagate_down[0]) {
    int count = 0;
    for (int i = 0; i < outer_num; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);

        if (has_ignore_label_ && label_value == ignore_label_) {
                continue;
        }

        Dtype p = std::max(
          bottom_data[i * dim + label_value * inner_num_ + j]
          , Dtype(kLOG_THRESHOLD));  

        bottom_diff[i * dim + label_value * inner_num_ + j] = Dtype(1.0) / p;
        ++count;
      }
    }
    Dtype scale = - top[0]->cpu_diff()[0] / (std::max(Dtype(1.0), Dtype(count)));
    caffe_scal(bottom[0].count(), scale, bottom_diff);
  }

  if (propagate_down[2]) {
    int count = 0;
    for (int i = 0; i < outer_num; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);

        if (has_ignore_label_ && label_value == ignore_label_) {
                continue;
        }

        Dtype m = std::max(
          bottom_mask[i * inner_num_+ j]
          , Dtype(kLOG_THRESHOLD));  

        if (label_value != 0){
          mask_diff[i * inner_num_ + j] = Dtype(1.0) / m;
        }
        else{
          mask_diff[i * inner_num_ + j] = Dtype(1.0) / (m-1);
        }
        ++count;
      }
    }
    Dtype scale = - top[0]->cpu_diff()[0] / (std::max(Dtype(1.0), Dtype(count)));
    caffe_scal(bottom[2].count(), scale, mask_diff);
  }

}

INSTANTIATE_CLASS(MultinomialLogisticLossMaskLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticMaskLoss);

}  // namespace caffe
