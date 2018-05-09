//
//  Cross_Spilt.cpp
//  readcsv
//
//  Created by Lili Wang on 10/18/16.
//  Copyright © 2016 Lili Wang. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include<vector>
#include "Cross_Spilt.hpp"
using namespace std;


Cross_Spilt::Cross_Spilt()
{
    datanumber=NULL;
    binlist=NULL;
    datanum=NULL;
}


Cross_Spilt::~Cross_Spilt()
{
    if(datanumber!=NULL) delete[] datanumber;
    if(binlist!=NULL) delete[] binlist;
    if(datanum!=NULL) delete [] datanum;
}


int* Cross_Spilt::coss(int *lable,int length,int cv_times)
{
    if(datanumber!=NULL) delete[] datanumber;
    if(binlist!=NULL) delete[] binlist;
    
    datanumber= new int[length];
    binlist=new int[cv_times];
    binlist[0]=0;
    int p,tmp;
    
    for (int i=0; i<=length; i++) datanumber[i]=i;
    for (int j=1; j<=cv_times; j++)
    {
        binlist[j]=length*(j)*1.0/cv_times;
        printf("j is %d index is %d\n",j,binlist[j]);
    }
    
    
    for (int i=length;i>=0;i--)
    {
        p=(int)(rand()/(1.0 * RAND_MAX)*length);
        tmp=datanumber[p];
        datanumber[p]=datanumber[i];
        datanumber[i]=tmp;
    }
    
    
    for (int i=0;i<=length; i++)
    {
        printf("%d\n",datanumber[i]);
        for (int j=0; j<cv_times; j++)
        {
            if (binlist[j+1]>=datanumber[i]&&datanumber[i]>=binlist[j]) {
                datanumber[i]=j;
                printf("i is %d  data is %d\n",i,datanumber[i]);
            }
        }
    }
    
    return datanumber;
}


void Cross_Spilt::spilt(int *lable,int length,float vatio,vector<int>  & train,vector<int>  & test)
{
    if(datanum!=NULL) delete [] datanum;
    datanum =new int[length];
    vector<int> lable0;
    vector<int> lable1;
    int p,tmp;
    
    for (int i=0; i<=length; i++) datanum[i]=i;
    for (int i=length;i>=0;i--)
    {
        p=(int)(rand()/(1.0 * RAND_MAX)*length);
        tmp=datanum[p];
        datanum[p]=datanum[i];
        datanum[i]=tmp;
    }
    
    
    for (int i=0; i<length; i++)
    {
        if (lable[i]==1)
        {
            lable0.push_back(datanum[i]);
        }
        else
        {
            lable1.push_back(datanum[i]);
        }

    }
    
    int leng0=(int)lable0.size();
    int leng1=(int)lable1.size();
    printf("leng0 is %d leng1 is %d\n",leng0,leng1);
    

    float rand_num0 = leng0*vatio;
    
    for (int i=0; i<leng0; i++) {
//        printf("i is %d,leng0 is %d,%f\n",i,leng0,rand_num0);
        if(i <=rand_num0)
        {
            train.push_back(lable0[i]);
        }
        else
        {
            test.push_back(lable0[i]);
        }
    }
    
    printf("traindata is %d, testdata is %d\n",(int)train.size(),(int)test.size());
    
    float rand_num1 = leng1*vatio;
    
    for (int i=0; i<leng1; i++) {
        
        if(i <=rand_num1)
        {
            train.push_back(lable1[i]);
        }
        else
        {
            test.push_back(lable1[i]);
        }
    }
    
    printf("traindata is %d, testdata is %d\n",(int)train.size(),(int)test.size());
}