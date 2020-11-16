//
//  Model.cpp
//  FxPred
//
//  Created by Liam Briegel on 13.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//

#include "Model.hpp"


/**
 *  Constructor of Model
 *
 *  takes care of initializing the layerns and so on
 */
Model::Model(int input_dim, int hidden_dim, int batch_size, int output_dim, int num_layers)
{
    // Init private Vars
    this->batch_size = batch_size;
    
    // Register layers
    this->lstm_layer = register_module("LSTM", torch::nn::LSTM(torch::nn::LSTMOptions(input_dim,hidden_dim)
                                                               .num_layers(num_layers).batch_first(false)
                                                               .bidirectional(true)));
    
    this->linear_layer = register_module("linear", torch::nn::Linear(hidden_dim * 2, hidden_dim * 2)); // * 2 because of bidirectionality
    
    this->output_layer = register_module("output", torch::nn::Linear(hidden_dim * 2, output_dim));
    
    
    // init the hidden
    this->hidden = std::make_tuple(torch::zeros({num_layers * 2, batch_size, hidden_dim}), torch::zeros({num_layers * 2, batch_size, hidden_dim})); // *2 because of bidirectionality
}


/**
 *  forwarding function
 *
 *  making the prediction. takes a tensor as argument
 */
torch::Tensor Model::forward(torch::Tensor input)
{
    // Feed through LSTM
    auto lstm_out = this->lstm_layer->forward(input, this->hidden);
    
    // save hidden state
    this->hidden = std::get<1>(lstm_out);
    
    // Feed through hidden linear layer
    auto output = this->linear_layer->forward( std::get<0>(lstm_out)[-1].view({batch_size, -1}) );
    
    // Feed through output layer and retrun
    return output_layer->forward(output);
}
