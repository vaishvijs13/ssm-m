#pragma once
#include "tensor.hpp"
#include "ops.hpp"
#include "broadcast.hpp"
#include "activations.hpp"

struct SelectiveScanFusedParams {
  std::shared_ptr<Tensor> A_log;
  std::shared_ptr<Tensor> log_dt;
  std::shared_ptr<Tensor> B;
  std::shared_ptr<Tensor> C;
  std::shared_ptr<Tensor> W_in;
  std::shared_ptr<Tensor> b_in;
};

//forward pass
inline std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> 
selective_scan_fused_forward(
  const std::shared_ptr<Tensor>& x,
  const SelectiveScanFusedParams& params,
  float log_dt_lo = -20.0f,
  float log_dt_hi = 5.0f
) {
  //validate inputs
  if (x->shape.size() != 2) throw std::runtime_error("selective_scan_fused: x must be [T, D_in]");
  if (!x->is_contiguous()) throw std::runtime_error("selective_scan_fused: x must be contiguous");
  
  int T = x->shape[0];
  int D_in = x->shape[1];
  
  auto& A_log = params.A_log;
  auto& log_dt = params.log_dt;
  auto& B = params.B;
  auto& C = params.C;
  auto& W_in = params.W_in;
  auto& b_in = params.b_in;
  
  //check param shapes
  if (A_log->shape.size() != 1) throw std::runtime_error("A_log must be [D]");
  if (log_dt->shape.size() != 1) throw std::runtime_error("log_dt must be [D]");
  if (B->shape.size() != 1) throw std::runtime_error("B must be [D]");
  if (C->shape.size() != 1) throw std::runtime_error("C must be [D]");
  
  int D = A_log->shape[0];
  
  if (log_dt->shape[0] != D) throw std::runtime_error("log_dt shape mismatch");
  if (B->shape[0] != D) throw std::runtime_error("B shape mismatch");
  if (C->shape[0] != D) throw std::runtime_error("C shape mismatch");
  
  if (W_in->shape.size() != 2) throw std::runtime_error("W_in must be [D, D_in]");
  if (W_in->shape[0] != D || W_in->shape[1] != D_in) throw std::runtime_error("W_in shape mismatch");
  if (b_in->shape.size() != 1 || b_in->shape[0] != D) throw std::runtime_error("b_in shape mismatch");
  
  if (!A_log->is_contiguous() || !log_dt->is_contiguous() || !B->is_contiguous() || 
      !C->is_contiguous() || !W_in->is_contiguous() || !b_in->is_contiguous()) {
    throw std::runtime_error("selective_scan_fused: all params must be contiguous");
  }
  
  vector_float y_data(T, 0.0f);
  
  //hidden state buffer
  vector_float h_data((Size)T * (Size)D, 0.0f);
  
  //scratch buffers
  vector_float u(D, 0.0f);
  vector_float h_prev(D, 0.0f);
  
  //pre-compute discretization params
  vector_float A_discrete(D);
  vector_float dt_vals(D);
  vector_float decay_vals(D);
  
  for (int d = 0; d < D; d++) {
    float A = -softplus_f(A_log->flat(d));
    float log_dt_val = log_dt->flat(d);
    
    if (log_dt_val < log_dt_lo) log_dt_val = log_dt_lo;
    if (log_dt_val > log_dt_hi) log_dt_val = log_dt_hi;
    
    float dt = std::exp(log_dt_val);
    float decay = std::exp(A * dt);
    
    A_discrete[d] = A;
    dt_vals[d] = dt;
    decay_vals[d] = decay;
  }
  
  for (int t = 0; t < T; t++) {
    for (int d = 0; d < D; d++) {
      float acc = b_in->flat(d);
      for (int j = 0; j < D_in; j++) {
        acc += W_in->flat((Size)d * D_in + j) * x->flat((Size)t * D_in + j);
      }
      u[d] = acc;
    }
    
    //fused SSM update and output accumulation
    float y_t = 0.0f;
    
    for (int d = 0; d < D; d++) {
      float decay = decay_vals[d];
      float bu = B->flat(d) * u[d];
      
      float h_new = decay * h_prev[d] + (1.0f - decay) * bu;
      
      h_data[(Size)t * D + d] = h_new;
      
      y_t += C->flat(d) * h_new;
      
      h_prev[d] = h_new;
    }
    
    y_data[t] = y_t;
  }
  
  //output tensors
  bool req = track_grad(x, A_log, log_dt, B, C, W_in, b_in);
  auto y = std::make_shared<Tensor>(std::move(y_data), vector_int{T, 1}, req);
  auto h_all = std::make_shared<Tensor>(std::move(h_data), vector_int{T, D}, false);
  
  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {x, A_log, log_dt, B, C, W_in, b_in};
    
    auto h_all_ptr = h_all;
    
    y->node->backward = [=](const vector_float& gy) -> std::vector<vector_float> {
      std::vector<vector_float> grads(7);
      grads[0].assign(x->numel(), 0.0f);
      grads[1].assign(A_log->numel(), 0.0f);
      grads[2].assign(log_dt->numel(), 0.0f);
      grads[3].assign(B->numel(), 0.0f);
      grads[4].assign(C->numel(), 0.0f);
      grads[5].assign(W_in->numel(), 0.0f);
      grads[6].assign(b_in->numel(), 0.0f);
      
      //gradient
      vector_float dh(D, 0.0f);
      
      //backward scan (rev)
      for (int t = T - 1; t >= 0; t--) {
        //gradient from output
        for (int d = 0; d < D; d++) {
          dh[d] += gy[t] * C->flat(d);
          
          if (C->req_grad) {
            grads[4][d] += gy[t] * h_all_ptr->flat((Size)t * D + d);
          }
        }
      
        vector_float du(D, 0.0f);
        
        for (int d = 0; d < D; d++) {
          float h_t = h_all_ptr->flat((Size)t * D + d);
          float h_prev_val = (t > 0) ? h_all_ptr->flat((Size)(t-1) * D + d) : 0.0f;
          
          //recompute forward quantities
          float A = -softplus_f(A_log->flat(d));
          float log_dt_val = log_dt->flat(d);
          if (log_dt_val < log_dt_lo) log_dt_val = log_dt_lo;
          if (log_dt_val > log_dt_hi) log_dt_val = log_dt_hi;
          float dt = std::exp(log_dt_val);
          float decay = std::exp(A * dt);
          
          float u_val = b_in->flat(d);
          for (int j = 0; j < D_in; j++) {
            u_val += W_in->flat((Size)d * D_in + j) * x->flat((Size)t * D_in + j);
          }
          
          float bu = B->flat(d) * u_val;
          float dh_prev = dh[d] * decay;
          du[d] = dh[d] * (1.0f - decay) * B->flat(d);
          if (B->req_grad) {
            grads[3][d] += dh[d] * (1.0f - decay) * u_val;
          }
          float d_decay = dh[d] * (h_prev_val - bu);
          
          float d_dt = d_decay * A * decay;
          
          if (log_dt->req_grad && log_dt_val >= log_dt_lo && log_dt_val <= log_dt_hi) {
            grads[2][d] += d_dt * dt;
          }
          
          float dA = d_decay * dt * decay;
          
          if (A_log->req_grad) {
            float sig = sigmoid_f(A_log->flat(d));
            grads[1][d] += dA * (-sig);
          }
          
          //propagate gradient to prev timestep
          if (t > 0) {
            dh[d] = dh_prev;
          } else {
            dh[d] = 0.0f;
          }
        }
        
        //backprop
        for (int d = 0; d < D; d++) {
          if (b_in->req_grad) {
            grads[6][d] += du[d];
          }
          
          if (W_in->req_grad || x->req_grad) {
            for (int j = 0; j < D_in; j++) {
              if (W_in->req_grad) {
                grads[5][(Size)d * D_in + j] += du[d] * x->flat((Size)t * D_in + j);
              }
              if (x->req_grad) {
                grads[0][(Size)t * D_in + j] += du[d] * W_in->flat((Size)d * D_in + j);
              }
            }
          }
        }
      }
      
      return grads;
    };
  }
  
  auto h_final = std::make_shared<Tensor>(
    h_all->storage,
    (Size)(T - 1) * D,
    vector_int{D},
    contiguous_strides(vector_int{D}),
    false
  );
  
  return {y, h_final};
}

//helper
inline bool track_grad(
  const std::shared_ptr<Tensor>& a,
  const std::shared_ptr<Tensor>& b,
  const std::shared_ptr<Tensor>& c,
  const std::shared_ptr<Tensor>& d,
  const std::shared_ptr<Tensor>& e,
  const std::shared_ptr<Tensor>& f,
  const std::shared_ptr<Tensor>& g
) {
  return grad_enabled() && (
    a->req_grad || b->req_grad || c->req_grad || d->req_grad || 
    e->req_grad || f->req_grad || g->req_grad
  );
}
