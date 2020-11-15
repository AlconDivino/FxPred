//
//  DataManager.cpp
//  FxPred
//
//  Created by Liam Briegel on 12.11.20.
//  Copyright © 2020 Liam Briegel. All rights reserved.
//

#include "DataManager.hpp"


/**
 *  Takes filepath to load candles into a std::vector<candle> and returns it.
 *
 *  In file not valid lines must be empty or marked with # or <
 *  File data must be organised in OHLC format. If in OHLCV the Volume is omitted.
 */
std::vector<candle> DataManager::loadCandles(std::string s_filepath)
{
    std::ifstream fs_open(s_filepath);
    std::string line;
    
    std::vector<candle> result;
    
    while( std::getline(fs_open, line))
    {
        // Skip non value lines
        if(line.empty() || line[0] == '#' || line[0] == '<')
            continue;
        
        // parse value into candle
        candle newCandle;
        std::stringstream ss_line(line);
        std::string fragment;
        
        for(int i = 0; std::getline(ss_line, fragment, ','); i++)
        {
            switch (i)
            {
                case 0:
                    newCandle.open = std::stof(fragment);
                    break;
                
                case 1:
                    newCandle.high = std::stof(fragment);
                    
                case 2:
                    newCandle.low = std::stof(fragment);
                    
                case 3:
                    newCandle.close = std::stof(fragment);
            }
        }
        
        // append to vector
        result.emplace_back(newCandle);
    }
    
    return result;
}


/**
 *  Splits src data into train and test vectors
 *
 *  the ratio declases which part is assigned to the training data set
 */
void DataManager::splitData(std::vector<candle> &src, float ratio, std::vector<candle> &train, std::vector<candle> &test)
{
    int idx = std::floor (static_cast<float>(src.size()) * ratio);
    
    train.clear();
    train.reserve(idx);
    for(int i = 0; i < idx; i++)
        train.push_back(src[i]);
    
    test.clear();
    test.reserve(src.size() - idx);
    for(int i = idx; i < src.size(); i++)
        test.push_back(src[i]);
}

