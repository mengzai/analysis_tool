#ifndef datacsv_hpp
#define datacsv_hpp

#include <stdio.h>


//-------------------------------------------------------------------------------------------------
class DataCsv
{
public:
    DataCsv();
    ~DataCsv();
    
    bool LoadCsv(const char *filename);
    int  GetRows();
    int  GetCols();
    
    const char* GetHead(int col); // get table head name
    float* GetCol(int col); // -1e10 = null
    float* GetRow(int row); // -1e10 = null
    int*   GetLab(); // get labels
    
private:
    int  rows;
    int  cols;
    int  *lab;
    float*mat;
    float*vec;
    char *head;
};

#endif
