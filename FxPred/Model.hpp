//
//  Model.hpp
//  FxPred
//
//  Created by Liam Briegel on 13.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//

#ifndef Model_hpp
#define Model_hpp

// C/C++
#include <stdio.h>

// External
#include <torch/torch.h>

// Project

class Model : public torch::nn::Module
{
    
public:
    Model(int input_dim, int hidden_dim, int batch_size, int output_dim, int num_layers);
    
    torch::Tensor forward(torch::Tensor input);
    
private:
    
    // hidden state
    std::tuple<torch::Tensor, torch::Tensor> hidden;
    
    // LSTM layer
    torch::nn::LSTM lstm_layer = NULL;
    torch::nn::Linear linear_layer = NULL;
    torch::nn::Linear output_layer = NULL;
    
};

#endif /* Model_hpp */
