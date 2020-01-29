 #include <stdio.h>

typedef unsigned int u32;
unsigned int NUM_ELEM = 6;


__host__ void cpu_sort(u32 * const data, const u32 num_elements)
{
	 u32 cpu_tmp_0[NUM_ELEM];
	 u32 cpu_tmp_1[NUM_ELEM];

	for (u32 bit=0; bit<32; bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for (u32 i=0; i<num_elements; i++)
		{
			const u32 d = data[i];
			const u32 bit_mask = (1 << bit);
			
			if ((d & bit_mask) > 0)
			{
				cpu_tmp_1[base_cnt_1] = d;
				base_cnt_1++;
			}
			else
			{
				cpu_tmp_0[base_cnt_0] = d;
				base_cnt_0++;
			}
		}

		//copy back back to source -then the zero list
		for(u32 i=0; i<base_cnt_0; i++)
		{
			data[i] = cpu_tmp_0[i];
		}

		//copy back back to source -then the one list
		for(u32 i=0; i<base_cnt_1; i++)
		{
			data[base_cnt_0+i] = cpu_tmp_1[i];
		}

	}
}

int main() {
	u32 DATA[NUM_ELEM] = {122, 10, 2, 22, 12, 9};
	cpu_sort(DATA, NUM_ELEM);

	for (u32 i=0;i<NUM_ELEM; i++)
	{
		printf(" %d ", DATA[i]);
	}
}
