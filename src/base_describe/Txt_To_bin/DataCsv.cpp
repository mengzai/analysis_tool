#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "datacsv.hpp"
//-------------------------------------------------------------------------------------------------
//构造函数
DataCsv::DataCsv()
{
    rows = cols = 0;
    lab = NULL;
    mat = NULL;
    vec = NULL;
    head = NULL;
}

//释放内存
//-------------------------------------------------------------------------------------------------
DataCsv::~DataCsv()
{
    if(lab!=NULL) delete[] lab;
    if(mat!=NULL) delete[] mat;
    if(vec!=NULL) delete[] vec;
    if(head!=NULL) delete[] head;
}


int DataCsv::GetRows()
{
    return rows;
}


int DataCsv::GetCols()
{
    return cols;
}


float* DataCsv::GetCol(int col){
    for(int j=0, k=col; j<rows; j++, k+=cols) {
        vec[j]=mat[k];
//        printf("j=%d,k=%d,%f\n",j,k,vec[j]);
    }
    return vec;
}


float* DataCsv::GetRow(int row)
{
    return mat+row*cols;
}


const char* DataCsv::GetHead(int col)
{
    int l=0,m=0;
    
    while (m!=col) {
        printf("l is %d m is %d head[l] is %c\n",l,m,head[l]);
        if (head[l]=='\0') {
            m++;
        }
        l++;
    }
    return head+l;
}


int*   DataCsv::GetLab()
{
    return lab;
}
//-------------------------------------------------------------------------------------------------
bool DataCsv::LoadCsv(const char *filename)
{
    rows = cols = 0;
    if(mat!=NULL) delete[] mat;
    if(vec!=NULL) delete[] vec;
    if(head!=NULL) delete[] head;
    
    FILE *pfile = fopen(filename, "rt");
    if(pfile==NULL){ printf("fopen(%s) failed\n", filename); return false; }
    
    // find how many rows and cols
    bool ret = true;
    const int len = 4096;
    char *buf = new char[len];
    while(fgets(buf, len, pfile)!=0)
    {
        rows++;
        
        int n = (int)strlen(buf), m = 0;
        for(int i=0; i<n; i++) if(buf[i]==',') m++;
        
        if(cols==0) cols = m;
        else if(m!=cols || buf[0]==',')
        {
//            printf("row %d is invalid\n", rows);
            ret = false;
            break;
        }
    }
    
    // load head and rows
    if(ret)
    {
        rows--; // not including head
//        printf("rows = %d, cols = %d\n", rows, cols);
        
        // load head
        fseek(pfile, SEEK_SET, 0);
        fgets(buf, len, pfile);
        printf("buf = %s\n", buf);
        
        int n = (int)strlen(buf);
//        printf("n = %d\n",n );
        head = new char[n+1];
        strcpy(head, buf);
        
        //change head , to '\0'
        int i=0;
        while (head[i]!='\0') {
            //        printf("%c",head[i]);
            if (head[i]==',') {
                head[i]='\0';
            }
            i++;
        }
        
        // load rows
        lab = new int[rows];
        mat = new float[rows*cols];
        vec = new float[rows>=cols?rows:cols];
        
        for(int i=0; i<rows && ret; i++)
        {
            fgets(buf, len, pfile);
//            printf("buf = %s", buf);
            
            float *mati = mat+i*cols;
            int n = (int)strlen(buf);
//            printf("n = %d\n", n);
            
            
            for(int j=0, l=0; j<n; j++)
            {
                // find ,
                int k = j;
                while(k<n && buf[k]!=',') k++;
//                printf("j = %d, k = %d, n = %d, %c\n", j, k, n, buf[k]);
                
                // load label
                if(j==0)
                {
                    if(sscanf(buf, "%d", lab+i)!=1) ret = false;
//                    printf("y = %d\n", lab[i]);
                    
                }
                
                // load attributes
                else
                {
                    
                    // ,
//                    printf("k is %d,j is %d,j+2 is %d\n",k,j,j+2);
                    if(k==j || (k==n && j+2==k && buf[n-1]=='\n')) mati[l] = -1e10; // null
                    // float,
                    else if(sscanf(buf+j, "%f", mati+l)!=1) ret = false;
//                                        printf("x = %f\n", mati[l]);
                    l++;
                }
                if(!ret)
                {
                                        printf("row %d is invalid\n", i);
                    break;
                }
                
                // update j
                j = k;
                if(k==n-1 && buf[k]==',') mati[l] = 1e-10; // if last row, no '\r\n'
            }
        }
    }
    delete[] buf;
    fclose(pfile);
    
    printf("LoadCsv(%s) %s\n", filename, ret ? "success" : "failure");
    return ret;
}

