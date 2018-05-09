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
	//��ʼ����
	for(int i=0; i<num; i++) shu[i] = i;

	//���ϴ��
	for(int i=num-1; i>0; i--)
	{
		//������0~i�У�ѡȡһ��Ԫ��
		int index = rand() % (i+1);

		//�ѵ�i��Ԫ�غ͵�index��Ԫ�ؽ���
		int temp = shu[i];
		shu[i] = shu[index];
		shu[index] = temp;
	}

	return shu;
}
