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



/** Main */
int main(int argc, const char * argv[])
{
    auto t = torch::zeros({1,1});
    
    std::cout << t << std::endl;
    
    auto ptr =(float*) t.data_ptr();
    
    *ptr = 1;
    
    std::cout << t << std:: endl;
    
    auto newt = t;
    
    *(float*)newt.data_ptr() = 2;
    
    std::cout << newt << std:: endl;
    
    std::cout << t << std::endl;
    
    return 0;
}
