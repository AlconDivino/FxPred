//
//  DataManager.hpp
//  FxPred
//
//  Created by Liam Briegel on 12.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//

#ifndef DataManager_hpp
#define DataManager_hpp


// C/C++
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

// External

// Project
#include "globals.h"

class DataManager
{
public:
    
    // Constructor
    DataManager();
    
    // Load candlesticks from file
    std::vector<candle> loadCandles(std::string s_filepath);
    
private:
    
    
};


#endif /* DataManager_hpp */
