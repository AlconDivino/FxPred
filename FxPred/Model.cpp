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
                                                               .num_layers(num_layers)
                                                               ));
    
    this->linear_layer = register_module("linear", torch::nn::Linear(hidden_dim , hidden_dim )); // * 2 because of bidirectionality
    
    this->output_layer = register_module("output", torch::nn::Linear(hidden_dim , output_dim));
    
    
    // init the hidden
    this->hidden = std::make_tuple(torch::zeros({num_layers , batch_size, hidden_dim}), torch::zeros({num_layers , batch_size, hidden_dim})); // *2 because of bidirectionality
}


/**
 *  forwarding function
 *
 *  making the prediction. takes a tensor as argument
 */
torch::Tensor Model::forward(torch::Tensor input)
{
    // Feed through LSTM
    auto lstm_out = this->lstm_layer->forward(input.clone(), this->hidden);
    
    // save hidden state
    this->hidden = std::get<1>(lstm_out);
    
    // Feed through hidden linear layer
    auto output = torch::sigmoid( this->linear_layer->forward( std::get<0>(lstm_out)[-1].view({batch_size, -1}) ) );
    
    // Feed through output layer and retrun
    return output_layer->forward(output);
}


/**
 *  Detach hidden because of the retrain graph issue
 *
 *  only detachtes the hidden states
 */
void Model::detachHidden()
{
    std::get<0>(hidden) = std::get<0>(hidden).detach();
    std::get<1>(hidden) = std::get<1>(hidden).detach();
}


/**
 *  Saving the two hiden states
 */
void Model::saveHidden(std::string s_filepath)
{ 
    torch::save(std::get<0>(hidden), s_filepath + "_hidden1.fxpred");
    torch::save(std::get<1>(hidden), s_filepath + "_hidden2.fxpred");
}


/**
 *  Loading the two hidden states
 */
void Model::loadHidden(std::string s_filepath)
{
    torch::load(std::get<0>(hidden), s_filepath + "_hidden1.fxpred");
    torch::load(std::get<1>(hidden), s_filepath + "_hidden2.fxpred");
}
