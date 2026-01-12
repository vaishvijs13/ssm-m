#pragma once
#include "tensor.hpp"

inline vector_int align_shape_right(const vector_int& shape, Size target_rank) {
  vector_int out;
  out.reserve(target_rank);
  Size pad = target_rank - shape.size();
  for (Size i = 0; i < pad; i++) out.push_back(1);
  for (int x : shape) out.push_back(x);
  return out;
}

inline vector_int broadcast_shape(const vector_int& a, const vector_int& b) {
  Size ra = a.size(), rb = b.size();
  Size r = (ra > rb) ? ra : rb;
  vector_int aa = align_shape_right(a, r);
  vector_int bb = align_shape_right(b, r);

  vector_int out(r, 1);
  for (Size i = 0; i < r; i++) {
    int da = aa[i], db = bb[i];
    if (da == db) out[i] = da;
    else if (da == 1) out[i] = db;
    else if (db == 1) out[i] = da;
    else throw std::runtime_error("broadcast_shape: incompatible dims");
  }
  return out;
}

//convert flat idx to multi-idnex
inline void flat_to_multi(Size flat, const vector_int& shape, const vector_int& strides, std::vector<int>& out_multi) {
  //shape + strides are same rank
  Size r = shape.size();
  out_multi.assign(r, 0);

  //divide by stride
  for (Size i = 0; i < r; i++) {
    int s = strides[i];
    out_multi[i] = static_cast<int>(flat / (Size)s);
    flat = flat % (Size)s;
  }
}

//convert back
inline Size multi_to_flat_aligned(const std::vector<int>& idx, const vector_int& aligned_shape, const vector_int& aligned_strides) {
  Size r = aligned_shape.size();
  Size off = 0;
  for (Size i = 0; i < r; i++) {
    off += (Size)idx[i] * (Size)aligned_strides[i];
  }
  return off;
}

inline vector_int aligned_strides_for_broadcast(const vector_int& input_shape, const vector_int& aligned_shape) {
  //aligned_shape is max rank shape
  vector_int input_strides = contiguous_strides(input_shape);

  Size r = aligned_shape.size();
  Size in_r = input_shape.size();
  Size pad = r - in_r;

  vector_int out_strides(r, 0);
  for (Size i = 0; i < r; i++) {
    int dim = aligned_shape[i];

    if (i < pad) {
      out_strides[i] = 0;
      continue;
    }

    //map to input axis
    Size in_axis = i - pad;
    int in_dim = input_shape[in_axis];

    //if input dim is 1, it broadcasts stride 0
    if (in_dim == 1) out_strides[i] = 0;
    else out_strides[i] = input_strides[in_axis];
  }
  return out_strides;
}

inline bool track_grad_bc(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { //if op tracks gradient
  return grad_enabled() && (a->req_grad || b->req_grad);
}
//ADD
inline std::shared_ptr<Tensor> add_bc(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  if (!a->is_contiguous() || !b->is_contiguous())
    throw std::runtime_error("add_bc: requires contiguous tensors (for now)");

  vector_int out_shape = broadcast_shape(a->shape, b->shape);
  Size out_numel = prod(out_shape);

  //align shapes to same rank
  Size r = out_shape.size();
  vector_int a_aligned_shape = align_shape_right(a->shape, r);
  vector_int b_aligned_shape = align_shape_right(b->shape, r);

  vector_int out_strides = contiguous_strides(out_shape);

  vector_int a_aligned_strides = aligned_strides_for_broadcast(a->shape, a_aligned_shape);
  vector_int b_aligned_strides = aligned_strides_for_broadcast(b->shape, b_aligned_shape);

  vector_float out(out_numel, 0.0f);
  std::vector<int> out_multi;

  for (Size i = 0; i < out_numel; i++) {
    flat_to_multi(i, out_shape, out_strides, out_multi);

    std::vector<int> a_idx = out_multi;
    std::vector<int> b_idx = out_multi;

    for (Size ax = 0; ax < r; ax++) {
      if (a_aligned_shape[ax] == 1) a_idx[ax] = 0;
      if (b_aligned_shape[ax] == 1) b_idx[ax] = 0;
    }

    Size a_off = multi_to_flat_aligned(a_idx, a_aligned_shape, a_aligned_strides);
    Size b_off = multi_to_flat_aligned(b_idx, b_aligned_shape, b_aligned_strides);

    out[i] = a->flat(a_off) + b->flat(b_off);
  }

  bool req = track_grad_bc(a, b);
  auto y = std::make_shared<Tensor>(std::move(out), out_shape, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {a, b};

    y->node->backward = [=](const vector_float& gout) {
      std::vector<vector_float> grads(2);
      grads[0].assign(a->numel(), 0.0f);
      grads[1].assign(b->numel(), 0.0f);

      std::vector<int> out_multi_local;
      for (Size i = 0; i < out_numel; i++) {
        flat_to_multi(i, out_shape, out_strides, out_multi_local);

        std::vector<int> a_idx = out_multi_local;
        std::vector<int> b_idx = out_multi_local;
        for (Size ax = 0; ax < r; ax++) {
          if (a_aligned_shape[ax] == 1) a_idx[ax] = 0;
          if (b_aligned_shape[ax] == 1) b_idx[ax] = 0;
        }

        Size a_off = multi_to_flat_aligned(a_idx, a_aligned_shape, a_aligned_strides);
        Size b_off = multi_to_flat_aligned(b_idx, b_aligned_shape, b_aligned_strides);

        if (a->req_grad) grads[0][a_off] += gout[i];
        if (b->req_grad) grads[1][b_off] += gout[i];
      }
      return grads;
    };
  }
  return y;
}

//MUL
inline std::shared_ptr<Tensor> mul_bc(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  if (!a->is_contiguous() || !b->is_contiguous())
    throw std::runtime_error("mul_bc: requires contiguous tensors (for now)");

  vector_int out_shape = broadcast_shape(a->shape, b->shape);
  Size out_numel = prod(out_shape);

  Size r = out_shape.size();
  vector_int a_aligned_shape = align_shape_right(a->shape, r);
  vector_int b_aligned_shape = align_shape_right(b->shape, r);

  vector_int out_strides = contiguous_strides(out_shape);

  vector_int a_aligned_strides = aligned_strides_for_broadcast(a->shape, a_aligned_shape);
  vector_int b_aligned_strides = aligned_strides_for_broadcast(b->shape, b_aligned_shape);

  vector_float out(out_numel, 0.0f);
  std::vector<int> out_multi;

  for (Size i = 0; i < out_numel; i++) {
    flat_to_multi(i, out_shape, out_strides, out_multi);

    std::vector<int> a_idx = out_multi;
    std::vector<int> b_idx = out_multi;
    for (Size ax = 0; ax < r; ax++) {
      if (a_aligned_shape[ax] == 1) a_idx[ax] = 0;
      if (b_aligned_shape[ax] == 1) b_idx[ax] = 0;
    }

    Size a_off = multi_to_flat_aligned(a_idx, a_aligned_shape, a_aligned_strides);
    Size b_off = multi_to_flat_aligned(b_idx, b_aligned_shape, b_aligned_strides);

    out[i] = a->flat(a_off) * b->flat(b_off);
  }

  bool req = track_grad_bc(a, b);
  auto y = std::make_shared<Tensor>(std::move(out), out_shape, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {a, b};

    y->node->backward = [=](const vector_float& gout) {
      std::vector<vector_float> grads(2);
      grads[0].assign(a->numel(), 0.0f);
      grads[1].assign(b->numel(), 0.0f);

      std::vector<int> out_multi_local;
      for (Size i = 0; i < out_numel; i++) {
        flat_to_multi(i, out_shape, out_strides, out_multi_local);

        std::vector<int> a_idx = out_multi_local;
        std::vector<int> b_idx = out_multi_local;
        for (Size ax = 0; ax < r; ax++) {
          if (a_aligned_shape[ax] == 1) a_idx[ax] = 0;
          if (b_aligned_shape[ax] == 1) b_idx[ax] = 0;
        }

        Size a_off = multi_to_flat_aligned(a_idx, a_aligned_shape, a_aligned_strides);
        Size b_off = multi_to_flat_aligned(b_idx, b_aligned_shape, b_aligned_strides);

        float a_val = a->flat(a_off);
        float b_val = b->flat(b_off);
        
        //chain rule
        if (a->req_grad) grads[0][a_off] += gout[i] * b_val;
        if (b->req_grad) grads[1][b_off] += gout[i] * a_val;
      }
      return grads;
    };
  }
  return y;
}
