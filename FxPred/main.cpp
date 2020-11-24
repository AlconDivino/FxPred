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
std::string filename = "../GBPJPY60-clean.csv"; //"/Users/Liam/XCodeProjects/Ressources/Datasets/EURUSD_H1_2017_2019-clean.csv";
int input_size = 150 * 4;
int output_size = 10 * 3;
int hidden_size = 60 * 4;
int num_layers = 10;
int batch_size = 1;
int maxEpoch = 10;

// global var
torch::Device device = torch::kCPU;




/** Main */
int main(int argc, const char * argv[])
{
    if(torch::cuda::is_available())
    {
        device = torch::kCUDA;
        printf("CUDA IS AVAILABLE");
    }
    
    // Load data from file
    DataManager manager;
    std::vector<candle> v_train;
    std::vector<candle> v_test;
    manager.splitData(manager.loadCandles(filename), 0.7, v_train, v_test);
    
    
    // Random for selection
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(input_size, (int)v_train.size() - output_size);
    int i_select = (int)v_train.size() - output_size;
    
    
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
    model->to(device);
    auto optim = torch::optim::Adam(model->parameters(), 0.01);
    
    
    // Training loop
    for(int epoch = 0; epoch < maxEpoch; epoch++)
    {
        for(size_t iter = 0; iter < v_train.size() - output_size; iter++)
        {
            model->zero_grad();
            
            // prepare data
            // Write it to the t_input and t_label directly via ptr
            //i_select = dist(generator);
            i_select++;
            if(i_select >= (int)v_train.size() -output_size)
                i_select = input_size;
            
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
            in.to(device);
            auto pred = model->forward( in );
            pred.to(device);
            auto loss = torch::mse_loss(pred, t_label.detach());
            
            // Loss print
            avgLoss += loss.item<float>();
            if(highestLoss < loss.item<float>())
                highestLoss = loss.item<float>();
            if(iter % 100 == 0 && iter != 0)
            {
                std::cout << "Iteration " << iter << ": Avg.Loss: " << avgLoss / 100 << " | CurrentLoss: " << loss.item<float>() << " | HighestLoss: " << highestLoss << std::endl;
                avgLoss = 0.;
                highestLoss = 0.;
            }
            
            // Optimize
            optim.zero_grad();
            loss.backward();
            optim.step();
            model->detachHidden();
            
        }
    }
    
    // Save the model and the  hidden tensors
    model->to(device);
    std::string s_saveName = std::to_string( std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1) );
    torch::save(model, s_saveName + "_model.fxpred");
    model->saveHidden(s_saveName);
    
    
    // **********************************************
    // Test the model
    // **********************************************
    printf("\nTesting the Mode\n\n");
    i_select = input_size;
    
    for(size_t iter = 0; iter < v_test.size(); iter++)
    {
        for(int i = i_select - input_size, idx = 0; i < i_select; i+=4, idx += 4)
        {
            p_input[idx] = v_test[i].open;
            p_input[idx+1] = v_test[i].high;
            p_input[idx+2] = v_test[i].low;
            p_input[idx+3] = v_test[i].close;
        }
        for(int i = i_select, idx = 0; i < i_select + output_size; i += 3, idx += 3)
        {
            p_label[idx] = v_test[i].high;
            p_label[idx+1] = v_test[i].low;
            p_label[idx+2] = v_test[i].close;
        }
    }
    
    
    return 0;
}
