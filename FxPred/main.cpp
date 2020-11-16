//
//  main.cpp
//  FxPred
//
//  Created by Liam Briegel on 12.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//


// C/C++
#include <iostream>
#include <random>

// External
#include <torch/torch.h>

// Project
#include "DataManager.hpp"
#include "Model.hpp"



// Global settings
std::string filename = "/Users/Liam/XCodeProjects/Ressources/Datasets/EURUSD_H1_2017_2019.csv";
int input_size = 30 * 4;
int output_size = 10 * 3;
int hidden_size = 60 * 4;
int num_layers = 5;
int batch_size = 1;




/** Main */
int main(int argc, const char * argv[])
{
    // Load data from file
    DataManager manager;
    std::vector<candle> v_train;
    std::vector<candle> v_test;
    manager.splitData(manager.loadCandles(filename), 0.7, v_train, v_test);
    
    
    // Random for selection
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(input_size, (int)v_train.size() - output_size);
    int i_select;
    
    
    // Create containers
    torch::Tensor t_input = torch::zeros({1, input_size});
    torch::Tensor t_label = torch::zeros({1, output_size});
    float *p_input = (float*)t_input.data_ptr();
    float *p_label = (float*)t_label.data_ptr();
    torch::Tensor pred;
    torch::Tensor loss;
    
    
    // Create model and optimizer
    auto model = std::make_shared<Model>(Model(input_size, hidden_size, batch_size, output_size, num_layers));
    auto optim = torch::optim::Adam(model->parameters(), 0.01);
    
    
    // Training loop
    for(size_t iter = 0; iter < 1000; iter++)
    {
        model->zero_grad();
        
        // prepare data
        // Write it to the t_input and t_label directly via ptr
        i_select = dist(generator);
        for(int i = i_select - input_size; i < i_select; i+=4)
        {
            p_input[i] = v_train[i].open;
            p_input[i+1] = v_train[i].high;
            p_input[i+2] = v_train[i].low;
            p_input[i+3] = v_train[i].close;
        }
        for(int i = i_select; i < i_select + output_size; i+=3)
        {
            p_label[i] = v_train[i].high;
            p_label[i+1] = v_train[i].low;
            p_label[i+2] = v_train[i].close;
        }
        
        // make prediction
        pred = model->forward(t_input.detach().clone().view({input_size, batch_size, -1}));
        auto loss = torch::mse_loss(pred, t_label.detach().clone());
        
        optim.zero_grad();
        loss.backward();
        optim.step();
        
        if(iter % 10 == 0)
            std::cout << "Iteration " << iter << ": Loss: " << loss.item<float>() << std::endl;
        
        
        
    }
    
    
    
    return 0;
}
