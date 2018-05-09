#include "shuffle.h"
#include <stdlib.h>
#include <time.h>


//-------------------------------------------------------------------------------------------------
CShuffle::CShuffle(int num)
{
	this->num = num;
	shu = new int[num];
}


//-------------------------------------------------------------------------------------------------
CShuffle::~CShuffle()
{
	delete[] shu;
}


//-------------------------------------------------------------------------------------------------
void CShuffle::SetRandomSeed()
{
	srand((unsigned int)time(0));
}


//-------------------------------------------------------------------------------------------------
int* CShuffle::RandomShuffle()
{
	//初始排列
	for(int i=0; i<num; i++) shu[i] = i;

	//随机洗牌
	for(int i=num-1; i>0; i--)
	{
		//从索引0~i中，选取一个元素
		int index = rand() % (i+1);

		//把第i个元素和第index个元素交换
		int temp = shu[i];
		shu[i] = shu[index];
		shu[index] = temp;
	}

	return shu;
}
