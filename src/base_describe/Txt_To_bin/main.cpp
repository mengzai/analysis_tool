#include <iostream>
#include "DataCsv.hpp"
#include "Cross_Spilt.hpp"
using namespace std;
int main(int argc, const char * argv[]) {

    DataCsv d;
    Cross_Spilt cros;
    d.LoadCsv("/Users/wanglili/Documents/workstation/code/c++/learn/xxd_test.csv");
    
    int clumns=d.GetCols();
    printf("clumns is %d\n",clumns);
    
    int Rowss=d.GetRows();
    printf("Rows is %d\n",Rowss);
    
//    float *di=d.GetCol(2);
//    printf("di is %f\n",*di);
    
//    float *ri=d.GetRow(1);
//    printf("ri is %f\n",*ri);
    
    int *lab=d.GetLab();
    printf("lab is %d\n",*lab);
    
//    const char *Head=d.GetHead(83);
//    printf("Head is %c\n",*Head);
    
//    int cv_times=5;
//    int *index=cros.coss(lab,Rowss,cv_times);
//    printf("index is %d\n",*index);
    
    float vatio=0.8;
    vector<int> train;
    vector<int> test;
    cros.spilt(lab,Rowss,vatio,train,test);
    cout<<train.size()<<endl;
//    cout<<nn.size()<<endl;
    
}
