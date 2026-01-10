#include <cmath>
#include <vector>
#include <random>
#include <iostream>

using Vec = std::vector<float>;
using Mat = std::vector<Vec>;

struct SelectiveSSM {
    int d_model, d_state;
    Mat A;
    Mat D;
    
    Mat W_delta, W_B, W_C;
    
    SelectiveSSM(int d_model, int d_state) : d_model(d_model), d_state(d_state) {
        std::mt19937 rng(42);
        std::normal_distribution<float> norm(0, 0.02f);
        
        auto init_mat = [&](int r, int c) -> Mat {
            Mat m(r, Vec(c));
            for (auto& row : m) for (auto& v : row) v = norm(rng);
            return m;
        };
        
        A = Mat(d_model, Vec(d_state));
        for (int i = 0; i < d_model; i++)
            for (int j = 0; j < d_state; j++)
                A[i][j] = -(j + 1);
        
        W_delta = init_mat(d_model, d_model);
        W_B = init_mat(d_model, d_state);
        W_C = init_mat(d_model, d_state);
        D = init_mat(d_model, 1);
    }
    
    void discretize(const Vec& delta, const Vec& A_row, Vec& A_bar, Vec& B_bar, const Vec& B) {
        for (int i = 0; i < d_state; i++) {
            float dA = delta[0] * A_row[i];
            A_bar[i] = std::exp(dA);
            B_bar[i] = (A_bar[i] - 1.0f) / A_row[i] * B[i];
        }
    }
    
    Vec forward(const Mat& x_seq) {
        int seq_len = x_seq.size();
        Vec h(d_state, 0.0f);
        Vec outputs;
        
        for (int t = 0; t < seq_len; t++) {
            const Vec& x = x_seq[t];
            
            float delta = std::log1p(std::exp(dot(W_delta[0], x)));
            Vec B(d_state), C(d_state);
            for (int i = 0; i < d_state; i++) {
                B[i] = dot(W_B[i], x);
                C[i] = dot(W_C[i], x);
            }
            
            Vec A_bar(d_state), B_bar(d_state);
            discretize({delta}, A[0], A_bar, B_bar, B);
            
            for (int i = 0; i < d_state; i++)
                h[i] = A_bar[i] * h[i] + B_bar[i] * x[0];
            
            float y = dot(C, h) + D[0][0] * x[0];
            outputs.push_back(y);
        }
        return outputs;
    }
    
    static float dot(const Vec& a, const Vec& b) {
        float s = 0;
        for (size_t i = 0; i < std::min(a.size(), b.size()); i++) s += a[i] * b[i];
        return s;
    }
};

int main() {
    SelectiveSSM ssm(64, 16);
    
    Mat seq(10, Vec(64, 0.1f));
    Vec out = ssm.forward(seq);
    
    std::cout << "out (top 5): ";
    for (int i = 0; i < 5; i++) std::cout << out[i] << " ";
    std::cout << "\n";
}
