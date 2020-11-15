//
//  main.cpp
//  FxPred
//
//  Created by Liam Briegel on 12.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//


// C/C++
#include <iostream>

// External
#include <torch/torch.h>

// Project
#include "DataManager.hpp"
#include "globals.h"
#include "Model.hpp"



// Global settings
std::string filename;
int input_size;
int output_size;
int hidden_size;
int num_layers;
int batch_size;




/** Main */
int main(int argc, const char * argv[])
{
    // Load data from file
    DataManager manager;
    auto v_data = manager.loadCandles(filename);
    
    // Create model
    auto model = std::make_shared<Model>(Model(input_size, hidden_size, batch_size, output_size, num_layers));
    
    
    
    
    return 0;
}
