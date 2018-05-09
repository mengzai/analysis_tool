#pragma once


//-------------------------------------------------------------------------------------------------
class CShuffle
{
public:
	CShuffle(int num);
	~CShuffle();
	
	void SetRandomSeed();
	int* RandomShuffle();

private:
	int  num;
	int *shu;
};
