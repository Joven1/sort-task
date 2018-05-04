/* To Build
   
*/

#include <immintrin.h> /* AVX intrinsics */
#include "avxcommon.h"
#include <iostream>
#include "avxsort_core.h"
#include <cstdio>
#include "utils.h"
using namespace std;

/**
 * Test all possible combinations for AVX cmp
 *
 * @param void
 */

void randomized_cmp_kv_pd(void)
{
	int j = 2;
	int k = 0;
	while(j==2)
	{
		double inA[4];
		double inAv[4];
		double inB[4];
		double inBv[4];
		double ret[4];
		for(int i = 0; i < 4; i++)
		{
			inA[i] = rand()%1000;
			inAv[i] = rand()%1000;
			inB[i] = rand()%1000;
			inBv[i]= rand()%1000;
			ret[i] = rand()%1000;
		}
		__m256d inA_r = _mm256_loadu_pd(inA);
		__m256d inAv_r = _mm256_loadu_pd(inAv);
		__m256d inB_r = _mm256_loadu_pd(inB);
		__m256d inBv_r = _mm256_loadu_pd(inBv);
		__m256d keyh, keyl, valh, vall;

		_mm256_cmp_kv_pd(inA_r, inAv_r, inB_r, inBv_r, valh, keyh, vall, keyl);


		for(int i = 0; i < 3; i++)
		{
			if(((valh[i] == vall[i])&&(keyl[i]>keyh[i])))
			{
				j = 1;
				_mm256_storeu_pd(ret, inA_r);
				dump_arr_double("inA_r", ret);
				_mm256_storeu_pd(ret, inB_r);
				dump_arr_double("inB_r", ret);
				_mm256_storeu_pd(ret, valh);
				dump_arr_double("valh", ret);
				_mm256_storeu_pd(ret, vall);
				dump_arr_double("vall", ret);
				
				
				_mm256_storeu_pd(ret, inAv_r);
				dump_arr_double("inAv_r", ret);
				_mm256_storeu_pd(ret, inBv_r);
				dump_arr_double("inBv_r", ret);
				_mm256_storeu_pd(ret, keyh);
				dump_arr_double("keyh", ret);
				_mm256_storeu_pd(ret, keyl);
				dump_arr_double("keyl", ret);
				break;
			}
			else if(valh[i] < vall[i])
			{
				printf("ERROR IS FOUND\n");
				_mm256_storeu_pd(ret, inA_r);
				dump_arr_double("inA_r", ret);
				_mm256_storeu_pd(ret, inAv_r);
				dump_arr_double("inAv_r", ret);
				_mm256_storeu_pd(ret, inB_r);
				dump_arr_double("inB_r", ret);
				_mm256_storeu_pd(ret, inBv_r);
				dump_arr_double("inBv_r", ret);
				_mm256_storeu_pd(ret, keyh);
				dump_arr_double("keyh", ret);
				_mm256_storeu_pd(ret, valh);
				dump_arr_double("valh", ret);
				_mm256_storeu_pd(ret, keyl);
				dump_arr_double("keyl", ret);
				_mm256_storeu_pd(ret, vall);
				dump_arr_double("vall", ret);
				j = 1;
			}
		}
	
}
}
	



/**
 * Test singular combination for AVX cmp
 *
 * @param void
 */

void test_cmp_kv_pd(void)
{
    double inA[4] = {1, 2, 3, 4};
    double inAv[4] = {5, 7, 6, 2};
    double inB[4] = {11, 12, 13, 14};
    double inBv[4] = {3, 9, 8, 4};

    double ret[4];
    __m256d inA_r = _mm256_loadu_pd(inA);
    __m256d inAv_r = _mm256_loadu_pd(inAv);
    __m256d inB_r = _mm256_loadu_pd(inB);
    __m256d inBv_r = _mm256_loadu_pd(inBv);
    __m256d keyh, keyl, valh, vall;

    _mm256_storeu_pd(ret, inA_r);
    dump_arr_double("inA_r", ret);
    _mm256_storeu_pd(ret, inAv_r);
    dump_arr_double("inAv_r", ret);
    _mm256_storeu_pd(ret, inB_r);
    dump_arr_double("inB_r", ret);
    _mm256_storeu_pd(ret, inBv_r);
    dump_arr_double("inBv_r", ret);

    _mm256_cmp_kv_pd(inA_r, inAv_r, inB_r, inBv_r, keyh, valh, keyl, vall);

    _mm256_storeu_pd(ret, keyh);
    dump_arr_double("keyh", ret);
    _mm256_storeu_pd(ret, valh);
    dump_arr_double("valh", ret);
    _mm256_storeu_pd(ret, keyl);
    dump_arr_double("keyl", ret);
    _mm256_storeu_pd(ret, vall);
    dump_arr_double("vall", ret);
}


/**
 * Test combinations for AVX inregister sort
 *
 * @param void
 */
void test_inregister_sort(void)
{
    int64_t in[16] =  { 4, 12, 345, 54,    2,  534,  24,  24, 132,  23, 123,  14, 14,  41, 53};
    int64_t inv[16] = {13, 43,  34, 12, 2352, 2431, 423, 143, 124, 234, 134, 124, 14, 414, 46};
    int64_t out[16];
    int64_t outv[16];
    int64_t out1[16];
    int64_t outv1[16];
    inregister_sort_keyval32(in, inv, out, outv);
    dump_arr_int64("out", out, 16);
    dump_arr_int64("outv", outv, 16);

}

#define LENA 100
#define LENB 100
void test_merge16_varlen(void)
{
    int64_t inA[LENA] =  {44,  49,  70,  74,  82,  108,  115,  130,  163,  165,  180,  185,  187,  190,  195,  203,  225,  234,  273,  303,  320,  330,  333,  339,  341,  345,  349,  350,  371,  373,  381,  404,  406,  407,  425,  432,  435,  445,  453,  458,  469,  501,  517,  521,  536,  557,  558,  561,  565,  591,  599,  603,  613,  619,  621,  629,  639,  644,  649,  650,  662,  667,  679,  680,  683,  700,  704,  706,  707,  710,  722,  723,  724,  725,  725,  728,  730,  739,  742,  769,  773,  774,  777,  807,  838,  863,  866,  879,  884,  889,  895,  895,  919,  934,  942,  946,  951,  960,  960,  980};
    int64_t inAv[LENA] = {44,  49,  70,  74,  82,  108,  115,  130,  163,  165,  180,  185,  187,  190,  195,  203,  225,  234,  273,  303,  320,  330,  333,  339,  341,  345,  349,  350,  371,  373,  381,  404,  406,  407,  425,  432,  435,  445,  453,  458,  469,  501,  517,  521,  536,  557,  558,  561,  565,  591,  599,  603,  613,  619,  621,  629,  639,  644,  649,  650,  662,  667,  679,  680,  683,  700,  704,  706,  707,  710,  722,  723,  724,  725,  725,  728,  730,  739,  742,  769,  773,  774,  777,  807,  838,  863,  866,  879,  884,  889,  895,  895,  919,  934,  942,  946,  951,  960,  960,  980};
    int64_t inB[LENB] = {18,  28,  60,  68,  79,  80,  86,  89,  96,  129,  170,  179,  180,  202,  202,  221,  225,  227,  234,  235,  287,  288,  309,  314,  320,  335,  337,  343,  343,  345,  357,  357,  362,  390,  399,  400,  414,  414,  421,  445,  454,  456,  484,  510,  511,  522,  522,  532,  551,  561,  574,  585,  594,  609,  618,  628,  634,  639,  643,  683,  698,  700,  705,  736,  758,  762,  774,  775,  779,  799,  805,  811,  814,  816,  820,  841,  845,  847,  847,  851,  865,  874,  875,  879,  896,  897,  903,  908,  922,  927,  935,  942,  956,  962,  969,  970,  977,  981,  988,  997};
    int64_t inBv[LENB] = {18,  28,  60,  68,  79,  80,  86,  89,  96,  129,  170,  179,  180,  202,  202,  221,  225,  227,  234,  235,  287,  288,  309,  314,  320,  335,  337,  343,  343,  345,  357,  357,  362,  390,  399,  400,  414,  414,  421,  445,  454,  456,  484,  510,  511,  522,  522,  532,  551,  561,  574,  585,  594,  609,  618,  628,  634,  639,  643,  683,  698,  700,  705,  736,  758,  762,  774,  775,  779,  799,  805,  811,  814,  816,  820,  841,  845,  847,  847,  851,  865,  874,  875,  879,  896,  897,  903,  908,  922,  927,  935,  942,  956,  962,  969,  970,  977,  981,  988,  997};
    int64_t out[LENA + LENB];
    int64_t outv[LENA + LENB];

    merge16_varlen(inA, inAv, inB, inBv, out, outv, LENA, LENB);
    dump_arr_int64("out", out, LENA + LENB);
    dump_arr_int64("outv", outv, LENA + LENB);
}


void
avxsort_unaligned(int64_t ** inputptr, int64_t ** inputptrv, 
                  int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems);
void
stabilize_unaligned(int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems);


/**
 * Test avx sort
 *
 * @param void
 */
void test_avxsort_unaligned();



/**
 * Test avxsort_block 
 *
 * @param void
 */
void test_blocksize();

/**
 * Test input/output lists created
 *
 */
void test_input_lists(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,int list_length);

void test_output_lists(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,int list_length);


/**
 * Print input/output lists created
 *
 */
void print_output(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out);

void print_input(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out);


/**
 * Find size of sorted elements in array
 *
 */
void test_sorted_list_size(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,uint64_t nitems);

/**
 * Randomized test for stability
 *
 */
void test_sort();


#define NUM_ITEMS (10000000000000000)  /* 2 * L2_CACHE_SIZE */
#include <time.h>

int main(int argc, char *argv[])
{
	int * ptr;
	printf("%zu\n",sizeof(ptr));
		test_sort();
        return 0;
}

void test_blocksize()
{
	int64_t * input, * keys_in, *output, *keys_out;
	int count = 0;
	input = (int64_t*)malloc(BLOCKSIZE* sizeof (int64_t));
	output = (int64_t*)malloc(BLOCKSIZE * sizeof (int64_t));
	keys_in = (int64_t*)malloc(BLOCKSIZE * sizeof (int64_t));
	keys_out = (int64_t*)malloc(BLOCKSIZE * sizeof (int64_t));
	for(int i = 0; i < BLOCKSIZE;i++)
	{
		input[i] = rand()%100000;
		keys_in[i] = i;
	}
	avxsort_block(&input,&keys_in, &output, &keys_out, BLOCKSIZE);
	//print_output(input,keys_in,output,keys_out);
	//print_input(input,keys_in,output,keys_out);
	test_input_lists(input,keys_in,output,keys_out,BLOCKSIZE/2);
	test_output_lists(input,keys_in,output,keys_out,BLOCKSIZE);
	//test_sorted_list_size(input,keys_in,output,keys_out,(uint64_t)BLOCKSIZE);
	int errorcount = 0;
	for(int i = 0; i < BLOCKSIZE;i++)
	{
		if((output[i] == output[i+1])&&(keys_out[i] > keys_out[i+1]))
		{
			printf("%d ERROR\n",i);	
		}
		printf("%" PRIu64 " ",output[i]);
		printf("%" PRIu64 "\n",keys_out[i]);
	}
}

void test_sort()
{
	int z = 1;
	double time_spent;
	srand(time(NULL));
	clock_t start;
	start = clock();
	FILE * fp1; 
	fp1 = fopen("items.txt","w");
	FILE * fp2;
	fp2 = fopen("time.txt","w");
	while(z==1)
	{
		printf("yes\n");
	for(int64_t k = 1; k < NUM_ITEMS; k=2*k)
	{
		
		//printf("K: %" PRIu64 "\n",k);
		int64_t * input, * keys_in, *output, *keys_out;
		int count = 0;
		input = (int64_t*)malloc(k * sizeof (int64_t));
		output = (int64_t*)malloc(k * sizeof (int64_t));
		keys_in = (int64_t*)malloc(k * sizeof (int64_t));
		keys_out = (int64_t*)malloc(k * sizeof (int64_t));
		for(int i = 0; i < k;i++)
		{
			input[i] = rand();
			keys_in[i] = i;
		}
		time_spent = (double) (clock()-start)/CLOCKS_PER_SEC;
		avxsort_unaligned(&input,&keys_in, &output, &keys_out, k);
		printf("Time: %f\n",time_spent);
		
		printf("The output size: %" PRIu64 "\n",k);
		fprintf(fp1,"%lf\n",time_spent);
		fprintf(fp2,"%" PRIu64 "\n",k);
		if(k >= 134217728*2)
		{
			fclose(fp1);
			fclose(fp2);
		}
		int errorcount = 0;
		for(int i = 0; i < k-1;i++)
		{
			if((output[i] == output[i+1])&&(keys_out[i] > keys_out[i+1]))
			{
				printf("%d ERROR ",i);	
				printf("%" PRIu64 " ",output[i]);
				printf("%" PRIu64 " ",keys_out[i]);
				printf("%" PRIu64 " ",output[i+1]);
				printf("%" PRIu64 "\n",keys_out[i+1]);
				errorcount++;
				z++;
			}
			if(output[i]>output[i+1])
			{
				printf("%d ERROR ",i);	
				printf("%" PRIu64 " ",output[i]);
				printf("%" PRIu64 " ",keys_out[i]);
				printf("%" PRIu64 " ",output[i+1]);
				printf("%" PRIu64 "\n",keys_out[i+1]);
				z++;
			}
		}
		printf("TOTAL ERRORS %d\n",errorcount);
		if(z == 2)
		{
			for(int i = 0; i < k; i++)
			{
				printf("%" PRIu64 " ",output[i]);
				printf("%" PRIu64 "\n",keys_out[i]);
			}
			break;
		}
		free(input);
		free(output);
		free(keys_in);
		free(keys_out);
	}
	}
}

void test_sorted_list_size(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,int64_t nitems)
{
	int64_t asdf = 0;
	asdf = 1;
	for(int i = 0; i < nitems ; i++)
	{
		asdf++;
	}
	printf("The output size: %" PRIu64 "\n",asdf-1);
}
void print_output(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out)
{
	int asdf = 0;
	for(int i = 0; i < BLOCKSIZE; i++)
	{
		printf("INPUT:  ");
		printf("%" PRIu64 " ",input[i]);
		printf("%" PRIu64 " ",keys_in[i]);
		printf("OUTPUT:  ");
		printf("%" PRIu64 " ",output[i]);
		printf("%" PRIu64 "\n",keys_out[i]);
		asdf++;
	}
}

void print_input(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out)
{
	int asdf = 0;
	for(int i = 0; i < BLOCKSIZE; i++)
	{
		printf("INPUT:  ");
		printf("%" PRIu64 " ",input[i]);
		printf("%" PRIu64 " ",keys_in[i]);
		printf("OUTPUT:  ");
		printf("%" PRIu64 " ",output[i]);
		printf("%" PRIu64 "\n",keys_out[i]);
		asdf++;
	}
}

void test_input_lists(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,int list_length)
{
	int k = 1;
	for(int i = 0; i < BLOCKSIZE;i+=list_length)
	{
		for(int j = 0; j < list_length-1; j++)
		{
			if((input[i+j]==input[i+k])&&(keys_in[i+j]>keys_in[i+k]))
			{
				printf("Index: %d ",i+j);
				printf("Element 1: %" PRIu64 " %" PRIu64 " Element 2: %" PRIu64 " %" PRIu64 "\n",input[i+j],keys_in[i+j],input[i+k],keys_in[i+k]);
			}
			k++;
		}
		k = 1;
	}
}

void test_output_lists(int64_t * input, int64_t * keys_in,int64_t *output,int64_t *keys_out,int list_length)
{
	int k = 1;
	for(int i = 0; i < BLOCKSIZE;i+=list_length)
	{
		for(int j = 0; j < list_length-1; j++)
		{
			if((output[i+j]==output[i+k])&&(keys_out[i+j]>keys_out[i+k]))
			{
				printf("Index: %d ",i+j);
				printf("Element 1: %" PRIu64 " %" PRIu64 " Element 2: %" PRIu64 " %" PRIu64 "\n",output[i+j],keys_out[i+j],output[i+k],keys_out[i+k]);
			}
			k++;
		}
		k = 1;
	}
}

void test_avxsort_unaligned()
{
		//local arrays
		int64_t * input, * keys_in, *output, *keys_out;
		int count = 0;
		input = (int64_t*)malloc(NUM_ITEMS * sizeof (int64_t));
		output = (int64_t*)malloc(NUM_ITEMS * sizeof (int64_t));
		keys_in = (int64_t*)malloc(NUM_ITEMS * sizeof (int64_t));
		keys_out = (int64_t*)malloc(NUM_ITEMS * sizeof (int64_t));
		for(int i = 0; i < NUM_ITEMS;i++)
		{
			input[i] = rand()%100;
			keys_in[i] = i;
		}
		avxsort_unaligned(&input,&keys_in,&output,&keys_out,NUM_ITEMS);
		/*
		for(int i = 0; i < NUM_ITEMS;i++)
		{
			printf("%" PRId64 "", output[i]);
			printf(" %" PRId64 "\n", keys_out[i]);
		}
		
		*/
		
}

void
avxsort_unaligned(int64_t ** inputptr, int64_t ** inputptrv,
	int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems)
{
	if (nitems <= 0)
		return;
	// dump_arr_int64("input", *inputptr, nitems);

	//dereference the addresses
	int64_t * input = *inputptr;
	int64_t * output = *outputptr;
	int64_t * inputv = *inputptrv;
	int64_t * outputv = *outputptrv;

	uint64_t i;
	
	//BLOCK SIZE == 16384 or 2^14 for some reason (found out, my comp supports only up to 2^14 cache) blockSize = l2cace / 2*sizeofint64
	uint64_t nchunks = (nitems / BLOCKSIZE);
	
	//this is the remainder after you divided up the blocks of data 2^14
	int rem = (nitems % BLOCKSIZE);

	//BREAKPOINT 1
	
	
	/* each chunk keeps track of its temporary memory offset */
	int64_t * ptrs[nchunks + 1][2];/* [chunk-in, chunk-out-tmp] */
	int64_t * ptrsv[nchunks + 1][2];/* [chunk-in, chunk-out-tmp] */
	uint32_t sizes[nchunks + 1];

	
	
	//create 2d array for pointers ptrs (values) and ptrsv (positions
	//ptrs is a 2d array which stores in the first column the first addresses of each data chunk
	//the second column stores the first addresses of the "output"
	//same difference with ptrsv
	//sizes just store the value 16384
	for (i = 0; i <= nchunks; i++) {
		ptrs[i][0] = input + i * BLOCKSIZE;
		ptrsv[i][0] = inputv + i * BLOCKSIZE;
		ptrs[i][1] = output + i * BLOCKSIZE;
		ptrsv[i][1] = outputv + i * BLOCKSIZE;
		sizes[i] = BLOCKSIZE;
	}
	
	
	
	

	/** 1) Divide the input into chunks fitting into L2 cache. */
	/* one more chunk if not divisible */
	//sort each block of data given along with the blocksize
	for (i = 0; i < nchunks; i++) {
		//given each address, sort the input (pass in the argument of itself)
		
		avxsort_block(&ptrs[i][0], &ptrsv[i][0], &ptrs[i][1], &ptrsv[i][1], BLOCKSIZE);
		// dump_arr_int64("check 0", ptrs[i][1], BLOCKSIZE);
		
		
		//simply swap the addresses (input -> output) (output -> input) between each of the rows
		//oh so thats why the input ptr has a value of zero after execution, it switches places with 
		//the output ptrs
		swap(&ptrs[i][0], &ptrs[i][1]);
		swap(&ptrsv[i][0], &ptrsv[i][1]);
	}

	
	//you sort the remaining block of data
	if (rem) {
		
		// xzl_bug_on(1);
		/* sort the last chunk which is less than BLOCKSIZE */
		
		avxsort_rem(&ptrs[i][0], &ptrsv[i][0], &ptrs[i][1], &ptrsv[i][1], rem);
		// dump_arr_int64("check 1", ptrs[i][1], rem);
		swap(&ptrs[i][0], &ptrs[i][1]);
		swap(&ptrsv[i][0], &ptrsv[i][1]);
		sizes[i] = rem;
	}

	
	
	/**
	* 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
	* obtain sorted sequences of length 2^{itr+1}.
	*/
	//if the remainder is greater than zero, you have additional chunks
	nchunks += (rem > 0);
	/* printf("Merge chunks = %d\n", nchunks); */
	const uint64_t logN = ceil(log2(nitems));
	
	//log2 blocksize = log2(BLOCKSIZE) and logN is log2(nitems)
	
	//hypothesis 3: you go from the log2(blocksize) to log2 of n items (rounded up)
	for (i = LOG2_BLOCKSIZE; i < logN; i++) {
		uint64_t k = 0;
		for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
			int64_t * inpA = ptrs[j][0];
			int64_t * inpB = ptrs[j + 1][0];
			int64_t * out = ptrs[j][1];
			int64_t * inpAv = ptrsv[j][0];
			int64_t * inpBv = ptrsv[j + 1][0];
			int64_t * outv = ptrsv[j][1];

			uint32_t  sizeA = sizes[j];
			uint32_t  sizeB = sizes[j + 1];

			/* need to change */
			merge16_varlen(inpA, inpAv, inpB, inpBv, out, outv, sizeA, sizeB);

			/* setup new pointers */
			ptrs[k][0] = out;
			ptrs[k][1] = inpA;
			ptrsv[k][0] = outv;
			ptrsv[k][1] = inpAv;

			sizes[k] = sizeA + sizeB;
			k++;
		}
		

		if ((nchunks % 2)) {
			/* just move the pointers */
			ptrs[k][0] = ptrs[nchunks - 1][0];
			ptrs[k][1] = ptrs[nchunks - 1][1];
			ptrsv[k][0] = ptrsv[nchunks - 1][0];
			ptrsv[k][1] = ptrsv[nchunks - 1][1];

			sizes[k] = sizes[nchunks - 1];
			k++;
		}

		nchunks = k;
	}
	/* finally swap input/output pointers, where output holds the sorted list */
	*outputptr = ptrs[0][0];
	*inputptr = ptrs[0][1];
	*outputptrv = ptrsv[0][0];
	*inputptrv = ptrsv[0][1];

}

void
stabilize_unaligned(int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems)
{
	uint64_t i = 0;
	//start at the beginning
	int64_t start = 0;
	uint64_t items = 0;
	uint64_t set = 1;
	i = 0;
	int64_t * outputptrv_start;
	int64_t * outputptr_start;
	outputptrv_start=*outputptrv;
	outputptr_start=*outputptr;
	
	int64_t ** outputptrv_index;
	int64_t ** outputptr_index;
	
	outputptrv_index = outputptrv;
	outputptr_index = outputptr;

	int64_t offset = 0;
	uint64_t current_items = 0;
	uint64_t first_index = 0;
	i = 0;
	set = 1;
/*
	while(i < nitems)
	{
		//compare the current value to its offset
		if(*(*(outputptr)) != *(*(outputptr)+offset))
		{
		}
		offset++;
		set++;
		i++;
	}
*/
	for(i = 0; i < NUM_ITEMS;i++)
	{
		/*
		printf("%" PRIu64 " \n",*(*outputptr));
		*outputptr = *outputptr + 1;
		*/
		if((*(*outputptr))!=(*(*outputptr+offset)))
		{
			avxsort_unaligned(outputptrv,outputptr,outputptrv,outputptr,offset);
			*outputptr = *outputptr+offset;
			*outputptrv = *outputptrv+offset;	
			offset = 0;
		}
		offset++;
	}
	avxsort_unaligned(outputptrv,outputptr,outputptrv,outputptr,offset);
	*outputptrv = outputptrv_start;
	*outputptr = outputptr_start;
	return;
}
