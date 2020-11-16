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
std::string filename = "/Users/Liam/XCodeProjects/Ressources/Datasets/EURUSD_H1_2017_2019-clean.csv";
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
    float *p_input = (float*)t_input[0].data_ptr();
    float *p_label = (float*)t_label.data_ptr();
    
    // Loss tracking
    float avgLoss = 0.;
    float highestLoss = 0.;
    
    
    // Create model and optimizer
    auto model = std::make_shared<Model>(Model(input_size, hidden_size, batch_size, output_size, num_layers));
    auto optim = torch::optim::Adam(model->parameters(), 0.01);
    
    
    // Training loop
    for(size_t iter = 0; iter < 100000; iter++)
    {
        model->zero_grad();
        
        // prepare data
        // Write it to the t_input and t_label directly via ptr
        i_select = dist(generator);
        for(int i = i_select - input_size, idx = 0; i < i_select; i+=4, idx += 4)
        {
            p_input[idx] = v_train[i].open;
            p_input[idx+1] = v_train[i].high;
            p_input[idx+2] = v_train[i].low;
            p_input[idx+3] = v_train[i].close;
        }
        for(int i = i_select, idx = 0; i < i_select + output_size; i += 3, idx += 3)
        {
            p_label[idx] = v_train[i].high;
            p_label[idx+1] = v_train[i].low;
            p_label[idx+2] = v_train[i].close;
        }
        
        // make prediction
        auto in = t_input.detach().view({1, 1, -1});
        auto pred = model->forward( in );
        auto loss = torch::mse_loss(pred, t_label.detach());
        
        // Loss print
        avgLoss += loss.item<float>();
        if(highestLoss < loss.item<float>())
            highestLoss = loss.item<float>();
        if(iter % 100 == 0 && iter != 0)
        {
            std::cout << "Iteration " << iter << ": Avg.Loss: " << avgLoss / 100 << " | CurrentLoss: " << loss.item<float>() << " | HighestLoss: " << highestLoss << std::endl;
            avgLoss = 0.;
        }
        
        // Optimize
        optim.zero_grad();
        loss.backward();
        optim.step();
        model->detachHidden();
        
    }
    
    
    // Save the model and the  hidden tensors
    std::string s_saveName = std::to_string( std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1) );
    torch::save(model, s_saveName + "_model.fxpred");
    model->saveHidden(s_saveName);
    
    
    return 0;
}
