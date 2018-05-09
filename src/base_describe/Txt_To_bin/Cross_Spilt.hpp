//
//  Cross_Spilt.hpp
//  readcsv
//
//  Created by Lili Wang on 10/18/16.
//  Copyright © 2016 Lili Wang. All rights reserved.
//

#ifndef Cross_Spilt_hpp
#define Cross_Spilt_hpp
#include<vector>
#include <stdio.h>
using namespace std;


#endif /* Cross_Spilt_hpp */
class Cross_Spilt
{
public:
    Cross_Spilt();
    ~Cross_Spilt();
    
    int* coss(int *lable,int rows,int cv_times);
    void spilt(int *lable,int length,float vatio,vector<int> & train,vector<int> & test);
    
private:
    int length;
    int *datanumber;
    int *binlist;
    int *datanum;
    vector<int> train;
    vector<int> test;
};
